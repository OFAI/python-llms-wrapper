"""Module implementing chatbot functionality for LLMs."""

import asyncio
import queue
import threading
import time
import random
import concurrent.futures
import logging

# Configure logging
# Using basicConfig will only configure handlers if they haven't been configured yet.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set a higher level for noisy async/threading debug logs if needed
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('threading').setLevel(logging.WARNING)


class ChatbotError(Exception):
    """Custom exception for chatbot errors."""
    pass

class FlexibleChatbot:
    """
    A chatbot class that processes messages asynchronously in a background thread
    and allows retrieving responses via either an async generator or a synchronous
    blocking method.
    """
    def __init__(self):
        # Queues for communication between the listening thread and the async loop thread.
        # asyncio.Queue is used because the processing loop runs on an asyncio loop.
        self._incoming_queue = asyncio.Queue()
        self._outgoing_queue = asyncio.Queue()

        # Event to signal the processing loop to stop. An asyncio.Event is needed
        # because the processing loop is async.
        self._stop_event = asyncio.Event()

        # Thread to run the asyncio event loop in the background.
        self._loop_thread: threading.Thread | None = None
        # The asyncio event loop instance running in _loop_thread.
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # The main asyncio task running the message processing logic.
        self._processing_task: asyncio.Task | None = None

        # Flag to indicate if the chatbot is intended to be running.
        self._running = False

        # Lock for the synchronous get_next_response method to prevent
        # multiple threads from trying to interact with the async loop simultaneously
        # via this method.
        self._get_response_lock = threading.Lock()

    def _run_loop_in_thread(self):
        """
        Target function for the dedicated background thread.
        It creates and runs the asyncio event loop.
        """
        logging.info("Asyncio loop thread starting.")
        # Create a new event loop specifically for this thread.
        self._event_loop = asyncio.new_event_loop()
        # Set this loop as the current loop for this thread.
        asyncio.set_event_loop(self._event_loop)

        try:
            # Create and schedule the main processing task on this loop.
            self._processing_task = self._event_loop.create_task(self._process_messages_loop())

            # Run the event loop until the stop event is set.
            # This blocks the thread until self._stop_event.set() is called from another thread.
            logging.info("Asyncio loop running until stop event...")
            # The loop waits here. Execution continues after stop_event is set.
            self._event_loop.run_until_complete(self._stop_event.wait())
            logging.info("Asyncio stop event received. Initiating shutdown sequence in loop thread.")

            # --- Graceful Shutdown Phase in Loop Thread ---
            # 1. Wait for the main processing task to finish (it exits its loop after stop event).
            #    This ensures messages currently being processed or already in the queue when
            #    stop was signalled get handled by _process_messages_loop's draining logic.
            logging.info("Waiting for processing task to finish after stop signal...")
            try:
                # Add a timeout for the processing task to complete its final cycle(s).
                # We use gather with return_exceptions=True to ensure we don't fail here
                # if the task raises an error during its final moments.
                self._event_loop.run_until_complete(asyncio.gather(asyncio.wait_for(self._processing_task, timeout=5.0), return_exceptions=True))
                logging.info("Processing task finished.")
            except asyncio.TimeoutError:
                 logging.warning("Processing task did not finish within timeout after stop signal.")
                 # If it didn't finish, try to cancel it as a fallback.
                 if not self._processing_task.done():
                     logging.warning("Cancelling processing task...")
                     try:
                         self._processing_task.cancel()
                         self._event_loop.run_until_complete(asyncio.gather(self._processing_task, return_exceptions=True))
                         logging.info("Processing task cancellation handled.")
                     except asyncio.CancelledError:
                         logging.info("Processing task cancellation handled.")
                     except Exception as e:
                         logging.exception(f"Error during processing task cancellation wait: {e}")
                 else:
                      logging.warning("Processing task was already done but wait_for timed out?")
            except Exception as e:
                 logging.exception(f"Error waiting for processing task to finish: {e}")


            # 2. Cancel any remaining tasks on this loop. This is crucial to clean up
            #    tasks created by run_coroutine_threadsafe (like pending queue.get calls
            #    from the synchronous get_next_response method).
            logging.info("Cancelling remaining tasks on the event loop...")
            pending_tasks = asyncio.all_tasks(self._event_loop)
            # Filter out the task running _run_loop_in_thread itself (which is done)
            # and the processing task (which should be done or cancelling).
            # We want to cancel other tasks, typically those waiting on queues/futures.
            tasks_to_cancel = [task for task in pending_tasks if task is not asyncio.current_task(self._event_loop) and not task.done()]

            if tasks_to_cancel:
                logging.info(f"Cancelling {len(tasks_to_cancel)} pending tasks.")
                # Gather all cancellation results, ignoring individual errors.
                self._event_loop.run_until_complete(asyncio.gather(*(task.cancel() for task in tasks_to_cancel), return_exceptions=True))
                # Now, wait for these cancelled tasks to actually complete their cancellation cleanup.
                self._event_loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logging.info("Pending tasks cancellation and waiting complete.")
            else:
                logging.info("No pending tasks found to cancel.")


            # 3. Add a small delay to allow any final cleanup logic (like in cancelled tasks) to run.
            logging.info("Giving a small moment for final cleanup...")
            try:
                 # Run a very short sleep on the loop.
                 self._event_loop.run_until_complete(asyncio.sleep(0.01))
            except Exception: # Catch potential errors if loop is already stopping/closed
                 pass


        except Exception as e:
             # Catch any exceptions that happened *during* the shutdown sequence itself.
             logging.exception(f"Exception during asyncio loop thread shutdown sequence: {e}")
        finally:
            # 4. Close the loop. This should happen only after tasks are cancelled and awaited.
            logging.info("Closing asyncio event loop.")
            # Check if loop is still open before closing
            if self._event_loop and not self._event_loop.is_closed():
                try:
                    self._event_loop.close()
                    logging.info("Asyncio event loop closed.")
                except Exception as e:
                    logging.exception(f"Error closing event loop: {e}")
            else:
                logging.warning("Attempted to close loop, but it was already None or closed.")

            # Clear references.
            asyncio.set_event_loop(None) # Unset loop for this thread
            self._event_loop = None
            self._processing_task = None # Ensure this is None
            logging.info("Asyncio loop thread finished.")


    async def _process_messages_loop(self):
        """
        Internal asyncio task coroutine that processes messages from the incoming queue.
        Handles errors during processing and puts responses/errors onto the outgoing queue.
        Gracefully drains the incoming queue during shutdown.
        """
        logging.info("Chatbot processing loop started.")
        # Loop as long as the stop event is not set OR there are items left in the incoming queue
        # to process. This allows draining the queue during shutdown.
        while not self._stop_event.is_set() or not self._incoming_queue.empty():
            try:
                # Get message:
                # If the stop event is NOT set, use await get() with a timeout.
                # If the stop event IS set, use get_nowait() to quickly process any remaining
                # items in the queue without blocking indefinitely. get_nowait() will raise
                # QueueEmpty when the queue is fully drained during shutdown, allowing the
                # loop to exit gracefully.
                if not self._stop_event.is_set():
                     # Wait for the next message while not stopping.
                     # Use a small timeout to allow loop to check stop_event and do periodic tasks.
                     message, author = await asyncio.wait_for(self._incoming_queue.get(), timeout=0.1)
                     logging.info(f"Chatbot received: '{message}' from {author}'")
                else:
                     # If stopping, try to get existing messages without waiting.
                     # This will raise QueueEmpty when the queue is drained.
                     message, author = self._incoming_queue.get_nowait()
                     logging.info(f"Chatbot processing remaining queued: '{message}' from {author}'")

                # --- Chatbot Logic for processing a single message ---
                try:
                    await asyncio.sleep(random.uniform(0.05, 0.2)) # Simulate processing time

                    response_text = None
                    if "hello" in message.lower():
                        response_text = f"Hi there, {author}! (async processed)"
                    elif "question" in message.lower():
                        logging.info("Offloading blocking I/O for question...")
                        try:
                            # Use asyncio.to_thread to run the blocking simulation in a separate thread.
                            # Errors from _simulate_blocking_io will be propagated by await.
                            response_text = await asyncio.to_thread(self._simulate_blocking_io, message)
                        except Exception as io_e:
                            logging.exception(f"Error during offloaded I/O for '{message}': {io_e}")
                            # On I/O error during processing, put an error message onto the outgoing queue.
                            response_text = f"Error processing question: {io_e}"

                    elif "error" in message.lower() and "processing" in message.lower():
                         # Simulate an internal processing error for this specific message.
                         raise ValueError("Simulated processing error for this message")

                    elif "io error" in message.lower():
                         # If message explicitly mentions "io error" but not "question", handle here
                         # or ensure it falls into the "question" path if that's where IO happens.
                         # Assuming it implies the question path for simulation.
                         if "question" not in message.lower():
                             response_text = f"Received potential I/O error trigger without question: '{message}'"
                         # Otherwise, the "question" path above will handle the IO error.

                    else:
                        pass # No response generated for other messages

                    if response_text:
                        logging.info(f"Chatbot generating response: '{response_text[:50]}...'")
                        # Put the generated response (or error message) onto the outgoing queue.
                        await self._outgoing_queue.put({"type": "message", "content": response_text, "author": "Chatbot"})

                except Exception as processing_e:
                    # Handle exceptions that occur *during* the processing of a single message.
                    logging.exception(f"Error processing message '{message}': {processing_e}")
                    # Put an error indicator/message onto the outgoing queue so consumers can see it.
                    await self._outgoing_queue.put({"type": "processing_error", "content": str(processing_e), "original_message": message})

                finally:
                    # Ensure task_done is called for the item retrieved from the incoming queue.
                    # This is crucial if asyncio.Queue.join() is used elsewhere to wait for processing completion.
                    self._incoming_queue.task_done()

            except asyncio.TimeoutError:
                # await get(timeout) expired. This happens when not stopping and the incoming queue is empty.
                # The loop condition allows us to just continue and check again.
                # This is where periodic actions would be triggered if needed.
                # logging.debug("Processing loop timed out waiting for message.")
                pass
            except asyncio.QueueEmpty:
                 # get_nowait() raised QueueEmpty. This should only happen when stopping
                 # (because we use await get() when not stopping). When it happens,
                 # it means the incoming queue is fully drained.
                 logging.debug("Incoming queue empty during stopping phase.")
                 # The while loop condition will be checked next. If stop_event is set,
                 # the loop will correctly terminate because the queue is also empty.

            except Exception as e:
                 # Catch any unexpected exceptions in the outer loop (e.g., issues with get_nowait, etc.).
                 logging.exception(f"Unexpected error in processing loop outer try block: {e}")
                 # If a critical error occurs that indicates the loop is broken, set the stop event
                 # to signal shutdown.
                 # self._stop_event.set() # Signal stop
                 # Add a small sleep here to prevent a tight error loop if the exception is continuous.
                 await asyncio.sleep(0.1)
                 # Depending on the severity, you might break the loop immediately:
                 # break


        logging.info("Chatbot processing loop finished.")


    def _simulate_blocking_io(self, message):
        """
        A synchronous method simulating blocking I/O (like a network request or disk access).
        Designed to be run in a separate thread (e.g., via asyncio.to_thread).
        """
        logging.info(f"  [Thread] Starting blocking I/O simulation for '{message[:20]}...'")
        time.sleep(random.uniform(0.5, 2.0)) # Simulate blocking I/O duration

        # Simulate an error sometimes based on message content
        if "io error" in message.lower():
             logging.error("  [Thread] Simulating I/O failure as requested.")
             raise IOError("Simulated I/O failure during processing")

        result = f"Processed blocking query for '{message[:20]}...'. Here's the info."
        logging.info(f"  [Thread] Blocking I/O simulation finished.")
        return result

    # --- Public Interface ---

    def listen(self, message: str, author: str):
        """
        Notify the chatbot about a new message. Can be called from any thread.
        Messages are queued for asynchronous processing. Non-blocking.
        """
        # Check if the chatbot's background loop is running and ready to receive messages.
        if not self.is_running() or self._event_loop is None or self._event_loop.is_closed():
            logging.warning(f"Chatbot not running or loop not ready. Message from {author} lost: '{message}'")
            # A more advanced version could buffer messages internally here until start() is called.
            return

        try:
            # Use call_soon_threadsafe to safely schedule the put operation onto the
            # chatbot's internal event loop thread. This is necessary because
            # asyncio.Queue methods are not thread-safe for calls from other threads.
            # put_nowait is used because listen() should not block the caller thread.
            # If the queue is full, QueueFull will be raised *in the loop thread*,
            # which should ideally be handled there (e.g., by logging a warning).
            self._event_loop.call_soon_threadsafe(
                self._incoming_queue.put_nowait, (message, author)
            )
            logging.debug(f"Message from {author} successfully queued.")

        except Exception as e:
             # Catch potential errors from call_soon_threadsafe itself (e.g., loop unexpectedly closed)
             logging.exception(f"Failed to enqueue message from {author} via call_soon_threadsafe: {e}")
             # The message might be lost here depending on error handling design choice.


    async def start(self):
        """
        Starts the internal chatbot processing thread and asyncio event loop.
        This method must be awaited and called from an asyncio context.
        """
        if self._running:
            logging.warning("Chatbot is already running.")
            return

        logging.info("Starting Chatbot...")
        # Reset event and flag before starting the thread.
        self._stop_event.clear()
        self._running = True

        # Create and start the thread that will run the asyncio loop.
        # Use daemon=True so the thread doesn't prevent the main program from exiting
        # if the main thread finishes before the chatbot is explicitly stopped.
        self._loop_thread = threading.Thread(target=self._run_loop_in_thread, daemon=True)
        self._loop_thread.start()

        # --- Wait for Loop Initialization ---
        # Wait briefly for the loop and processing task to be set up in the new thread.
        # This is a heuristic. A more robust approach involves the _run_loop_in_thread
        # signalling back when it's ready (e.g., using a threading.Event or Queue).
        # We need to wait long enough for _event_loop and _processing_task to be assigned.
        await asyncio.sleep(0.1)

        # Check if the loop and task were successfully initialized by the thread
        # and if the task is still running (not immediately done/errored).
        if self._event_loop is None or self._event_loop.is_closed() or \
           self._processing_task is None or self._processing_task.done():
             self._running = False # Mark as not running if initialization failed
             logging.error("Failed to initialize internal asyncio loop or processing task.")
             # Attempt to join the thread to clean up resources if it potentially failed immediately.
             if self._loop_thread and self._loop_thread.is_alive():
                 self._loop_thread.join(timeout=1) # Don't block the calling start indefinitely
             self._loop_thread = None
             self._event_loop = None # Ensure state is clean
             self._processing_task = None
             # Raise a specific error to indicate start failure.
             raise ChatbotError("Failed to initialize internal processing.")

        logging.info("Chatbot started successfully.")


    async def stop(self):
        """
        Signals the internal chatbot processing to stop gracefully.
        This method must be awaited and called from an asyncio context.
        It waits for the internal processing thread to finish.
        """
        # Check if the chatbot is in a state that can be stopped.
        if not self._running or self._loop_thread is None or not self._loop_thread.is_alive():
            logging.warning("Chatbot is not running or thread is not active.")
            # Clean up state just in case it's in an inconsistent state.
            self._running = False
            self._loop_thread = None
            self._event_loop = None
            self._processing_task = None
            return

        logging.info("Stopping Chatbot...")
        # Set the running flag to False immediately.
        self._running = False

        # Signal the processing loop to stop by setting the asyncio event.
        # This must be called thread-safely onto the internal loop thread.
        if self._event_loop and not self._event_loop.is_closed():
             try:
                # Schedule the event setting on the loop thread.
                self._event_loop.call_soon_threadsafe(self._stop_event.set)
                logging.debug("Stop event signalled thread-safely.")
             except Exception as e:
                 logging.exception(f"Error signalling stop event thread-safely: {e}")
                 # Even if signalling fails, proceed with trying to join the thread.
        else:
             logging.warning("Event loop not available or closed during stop signal. Cannot signal stop event.")
             # If the loop is gone, the thread might be exiting already, but cleanup might be missed.


        # Wait for the thread running the loop to join.
        # Use asyncio.to_thread to run the blocking .join() call in a separate thread
        # managed by asyncio, so it doesn't block the async loop calling stop().
        logging.info("Waiting for internal loop thread to join...")
        try:
            # Set a timeout for joining the thread in case it's stuck (e.g., in blocking I/O).
            # This prevents the stop() method from hanging indefinitely.
            await asyncio.to_thread(self._loop_thread.join, timeout=10.0) # Increased timeout to 10s

            # Check if the thread successfully joined or if the join timed out.
            if self._loop_thread and self._loop_thread.is_alive():
                 logging.error("Internal loop thread did not join within timeout!")
                 # Depending on requirements, you might log this and continue,
                 # or attempt more drastic measures (less recommended in libraries).
            else:
                 logging.info("Internal loop thread joined successfully.")

        except Exception as e:
             # Catch any exceptions during the asyncio.to_thread or join operation.
             logging.exception(f"Error waiting for internal loop thread to join: {e}")

        # Ensure state is completely reset regardless of join success/failure.
        self._loop_thread = None
        self._event_loop = None # Should be None after close in thread (or potentially None if join timed out)
        self._processing_task = None # Should be None after thread exits or cancellation

        logging.info("Chatbot stopped.")


    def is_running(self) -> bool:
        """
        Returns True if the chatbot's internal processing is intended to be active
        and the background thread is alive.
        """
        # Check the _running flag and the thread status for a more robust check.
        return self._running and self._loop_thread is not None and self._loop_thread.is_alive()


    # --- Speak Functionality Option 1: Async Generator ---
    async def responses(self):
        """
        An async generator that yields responses from the chatbot as they are ready.
        Consume this using 'async for'. Requires an asyncio event loop to be running
        in the context where this generator is iterated.
        """
        # Check if the chatbot is running before starting consumption.
        # Allow consumption if not running but queues aren't empty, to drain remaining items.
        if not self.is_running() and self._outgoing_queue.empty():
            logging.warning("Chatbot not running and outgoing queue is empty, responses generator yielding nothing.")
            return # Generator immediately stops if not running and nothing to drain.

        logging.info("Async responses generator started.")
        # Continue yielding as long as the chatbot is running OR there are items
        # left in the outgoing queue (to drain messages processed during shutdown).
        while self.is_running() or not self._outgoing_queue.empty():
             try:
                # Wait for a response with a short timeout.
                # The timeout allows the loop condition (self.is_running() and queue.empty()) to be checked
                # and prevents the generator from blocking indefinitely after stop is signalled
                # and the queue is empty.
                response = await asyncio.wait_for(self._outgoing_queue.get(), timeout=0.1) # Small timeout

                logging.debug(f"Async generator yielding response: {response}")
                yield response
                # Signal that the item has been consumed from the queue.
                # This must be done on the same loop the queue belongs to, which is the current loop here.
                self._outgoing_queue.task_done()

             except asyncio.TimeoutError:
                 # This exception occurs when get(timeout) expires before an item is available.
                 # The loop condition (while self.is_running() or not self._outgoing_queue.empty())
                 # will be checked next.
                 # logging.debug("Responses generator timed out waiting for response.")
                 pass # Continue the loop

             except Exception as e:
                # Catch any other exceptions during queue retrieval or yielding.
                logging.exception(f"Error in async responses generator while getting/yielding: {e}")
                # Decide how to handle error - yield an error message or break the generator?
                # Yielding an error message allows the consumer to handle it explicitly.
                yield {"type": "generator_error", "content": f"Error retrieving response: {e}"}
                # If the error is critical or unrecoverable for the generator, uncomment break:
                # break # Exit the generator loop on error.

        logging.info("Async responses generator finished.")


    # --- Speak Functionality Option 2: Synchronous Blocking Method ---
    def get_next_response(self, timeout: float | None = None):
        """
        Retrieves the next response, blocking until a response is available
        or the optional timeout occurs. Returns the response dictionary or None on timeout.
        Can be called from any synchronous thread, provided start() has been called.
        Raises ChatbotError if the chatbot is not running or an internal error occurs
        during retrieval. Returns None if the chatbot is stopping and the queue is empty.
        """
        # Acquire lock to ensure only one thread calls this method at a time (optional but safer)
        with self._get_response_lock:
            # Check if the chatbot's background loop is running and ready.
            # If not running but the outgoing queue is NOT empty, we still allow
            # attempting to get messages to drain the queue during sync shutdown.
            if not self.is_running() and self._outgoing_queue.empty():
                # If not running and queue is empty, there's nothing to get.
                # Avoid interacting with a potentially closed loop.
                 return None
            elif not self.is_running() and self._event_loop is None:
                 # If not running and loop is clearly gone, raise an error.
                 raise ChatbotError("Chatbot is not running and internal loop is not available.")
            elif not self.is_running() and self._event_loop.is_closed():
                 # If not running and loop is closed, also indicate error.
                 raise ChatbotError("Chatbot is not running and internal loop is closed.")


            # If we reach here, either running OR not running but queue has items.
            # In either case, we proceed to try and get from the queue via run_coroutine_threadsafe.
            try:
                # We need to run an async coroutine (self._outgoing_queue.get()) on the
                # chatbot's internal event loop from this external synchronous thread.
                # asyncio.run_coroutine_threadsafe is the standard and safe tool for this.
                # Note: asyncio.run_coroutine_threadsafe can raise RuntimeError if the loop is closed
                # or potentially other issues if called during a messy shutdown.
                coro = self._outgoing_queue.get()
                future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)

                # Wait for the future to complete in this thread (the calling synchronous thread),
                # with the specified timeout. This call blocks the current thread.
                # This can raise concurrent.futures.TimeoutError or exceptions propagated
                # from the coroutine (like exceptions put into the queue as error messages).
                response = future.result(timeout=timeout)

                # If we reached here, the .get() operation on the async queue was successful
                # and returned an item (response or error dict).
                # Now, signal task_done() back on the event loop thread, thread-safely.
                if self._event_loop and not self._event_loop.is_closed():
                    try:
                        # Schedule the task_done operation on the loop thread.
                        # Use call_soon_threadsafe to avoid potential race conditions
                        # if the loop is just about to close.
                        self._event_loop.call_soon_threadsafe(self._outgoing_queue.task_done)
                    except Exception as e:
                         # Log error if task_done cannot be scheduled (e.g., loop closed just now)
                         logging.exception(f"Error calling task_done thread-safely for {response}: {e}")
                else:
                     logging.warning(f"Event loop not available to call task_done for {response}.")


                logging.debug(f"Sync getter retrieved response: {response}")
                return response

            except concurrent.futures.TimeoutError:
                 # This exception occurs if future.result(timeout=...) expires.
                 # It means no item was available in the async queue within the timeout.
                 # This is a normal occurrence when the queue is empty.
                 # logging.debug("Sync getter timed out waiting for response.")
                 return None # Standard behavior for blocking get with timeout is returning None.

            except concurrent.futures.CancelledError:
                 # This could happen if the internal asyncio loop is stopping and cancels pending futures
                 # associated with run_coroutine_threadsafe. This is part of the graceful shutdown.
                 logging.warning("Sync get operation was cancelled (internal loop shutting down?).")
                 # Treat this as a non-critical event indicating shutdown or no more items.
                 # Return None to allow consumption loops to check if chatbot is still running.
                 return None

            except RuntimeError as e:
                 # Catch RuntimeErrors, specifically 'Event loop is closed', which can happen
                 # if run_coroutine_threadsafe is called right as the loop is closing.
                 if "Event loop is closed" in str(e):
                     logging.warning("Sync get called when event loop is closed.")
                     return None # Treat as no more items available due to shutdown
                 else:
                      logging.exception(f"Unexpected RuntimeError in sync get: {e}")
                      raise ChatbotError(f"Unexpected internal error during retrieval: {e}") from e # Re-raise as ChatbotError

            except Exception as e:
                # Catch any other unexpected exceptions during future.result retrieval or
                # exceptions propagated from the coroutine itself (like processing errors
                # if the consumer code didn't handle the dict).
                logging.exception(f"Error in synchronous get_next_response during future result: {e}")
                # Decide how to handle - re-raise as a ChatbotError
                raise ChatbotError(f"Error retrieving response: {e}") from e


# --- Example Usage ---

async def run_async_example():
    logging.info("\n" + "="*40 + "\n" + "--- Running Async Example ---" + "\n" + "="*40)
    chatbot = FlexibleChatbot()
    try:
        # Start the chatbot's background processing
        await chatbot.start()

        # Send messages to the chatbot (can be done from other threads/async tasks via .listen)
        logging.info("Sending initial messages in async example...")
        chatbot.listen("Hello chatbot!", "User1")
        await asyncio.sleep(0.01) # Give loop a moment to process listen call
        chatbot.listen("Tell me a story question?", "User2") # Blocking I/O simulation
        await asyncio.sleep(0.01)
        chatbot.listen("Another message.", "User1")
        await asyncio.sleep(0.01)
        chatbot.listen("Simulate IO error question with io error", "User3") # Should trigger IO error in to_thread
        await asyncio.sleep(0.01)
        chatbot.listen("Simulate processing error", "User4") # Should trigger processing error in logic
        await asyncio.sleep(0.01)
        chatbot.listen("One last message to ensure queue drains", "User5")
        await asyncio.sleep(0.01)


        # Consume responses using the async generator
        print("\n--- Responses (Async) ---")
        timeout_sec = 15 # seconds - Increased overall consumption timeout for example
        start_time = time.time()
        response_count = 0
        try:
            # Consume responses using the async generator interface.
            # The generator will yield responses as they become available or time out.
            async for response in chatbot.responses():
                print(f"Received (Async): {response}")
                response_count += 1

                # Example stopping condition: stop after an overall timeout for consumption.
                if time.time() - start_time > timeout_sec:
                    print("Stopping async consumption after overall timeout.")
                    break # Break the async for loop.

                # Example stopping condition: stop after receiving specific types of errors.
                # This correctly checks for the 'processing_error' and 'generator_error' types.
                if response.get("type") in ("processing_error", "generator_error"):
                    print("Stopping async consumption after receiving an error.")
                    # Allow processing/receiving any immediate follow-up messages if needed,
                    # then break the loop.
                    await asyncio.sleep(0.1) # Small delay to let logs/other things happen
                    break # Break the async for loop.

                # If you want to stop after a certain number of responses:
                # if response_count >= 6: # Example: Stop after 6 responses
                #     print(f"Stopping async consumption after receiving {response_count} responses.")
                #     break

        except Exception as e:
            # Catch any unexpected exceptions that occur within the async for loop itself.
            logging.exception("Exception during async response consumption:")

        print(f"Async consumption loop finished. Received {response_count} responses.")

    except ChatbotError as e:
        logging.error(f"Failed to start Chatbot: {e}")

    finally:
        # Ensure the chatbot is stopped gracefully even if errors occurred during consumption or start.
        if chatbot.is_running() or (chatbot._loop_thread and chatbot._loop_thread.is_alive()):
             logging.info("Async example: Chatbot appears to be running, attempting to stop.")
             # Add a timeout for the stop operation itself.
             try:
                 # Use asyncio.wait_for for the stop operation.
                 await asyncio.wait_for(chatbot.stop(), timeout=10)
                 logging.info("Async example: Chatbot stopped successfully.")
             except asyncio.TimeoutError:
                 logging.error("Async example: Chatbot stop operation timed out!")
             except Exception as e:
                 logging.exception(f"Async example: Error during chatbot stop: {e}")
        else:
             logging.info("Async example: Chatbot was not running or stopped unexpectedly.")

    logging.info("--- Async Example Finished ---")


def run_sync_example():
    logging.info("\n" + "="*40 + "\n" + "--- Running Sync Example ---" + "\n" + "="*40)
    chatbot = FlexibleChatbot()

    # Start the chatbot's internal async loop in its dedicated thread.
    # Since this function is synchronous, we use asyncio.run() just to await the start() method.
    # The internal loop then runs in its own thread, independent of the main thread here.
    logging.info("Starting Chatbot for sync use...")
    try:
        # Add a timeout for the start operation in case it hangs.
        asyncio.run(asyncio.wait_for(chatbot.start(), timeout=5))
        logging.info("Chatbot started successfully for sync use.")
    except (ChatbotError, asyncio.TimeoutError) as e:
        logging.error(f"Failed to start chatbot for sync use: {e}")
        # Ensure state is cleaned up if start failed
        if chatbot._loop_thread and chatbot._loop_thread.is_alive():
             logging.warning("Attempting to join thread after failed start.")
             chatbot._loop_thread.join(timeout=1)
        return # Cannot proceed if start failed


    # Send messages from the main synchronous thread
    logging.info("Sending initial messages in sync example...")
    chatbot.listen("Hello from Sync!", "SyncUser1")
    chatbot.listen("Sync question with io error", "SyncUser2") # Should trigger IO error
    chatbot.listen("Another sync message.", "SyncUser1")
    chatbot.listen("Sync error test?", "SyncUser3") # Should trigger processing error
    chatbot.listen("Final sync message.", "SyncUser1")


    # Consume responses using the synchronous method
    print("\n--- Responses (Sync) ---")
    messages_to_try_get = 8 # Try to get up to this many responses for demonstration
    received_count = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 3 # Stop after this many consecutive timeouts

    # Continue getting responses as long as the chatbot is running OR
    # there are items in the outgoing queue (to drain during shutdown) AND
    # we haven't exceeded maximum attempts AND haven't had too many consecutive timeouts.
    # The condition `chatbot.is_running()` allows the loop to stop naturally
    # when stop() is called externally, but we add timeout logic for robustness
    # if it stops producing messages unexpectedly.
    while (chatbot.is_running() or not chatbot._outgoing_queue.empty()) \
          and received_count < messages_to_try_get \
          and consecutive_timeouts < max_consecutive_timeouts:
        try:
            # Use a timeout for get_next_response(). This call will block the current thread.
            # If the queue becomes empty and no new messages are processed, this will time out.
            response = chatbot.get_next_response(timeout=2.0) # Timeout per get attempt

            if response:
                print(f"Received (Sync): {response}")
                received_count += 1
                consecutive_timeouts = 0 # Reset consecutive timeouts on success

                 # Example stopping condition based on content/type
                if response.get("type") in ("processing_error", "generator_error"):
                     print("Stopping sync consumption after receiving an error.")
                     # Decide if receiving an error should stop the consumption loop.
                     # break # Uncomment to stop on first error received.

                # Stop after a certain number of responses if needed:
                # if received_count >= 6: # Example: Stop after 6 responses
                #     print(f"Stopping sync consumption after receiving {received_count} messages.")
                #     break

            else:
                # response is None indicates a timeout occurred waiting for a message.
                print(f"Received (Sync): None (timeout after 2.0s).")
                consecutive_timeouts += 1
                # The loop condition checks max_consecutive_timeouts.

        except ChatbotError as e:
             # Catch errors raised by get_next_response itself (e.g., not running, internal retrieval error).
             logging.error(f"ChatbotError during sync get: {e}")
             break # Stop consumption loop on critical error from the method.
        except Exception as e:
             # Catch any other unexpected errors during the get loop.
             logging.exception("Unexpected error during sync get loop:")
             break # Stop consumption loop on unexpected error.

    print(f"Sync consumption loop finished. Received {received_count} responses. Consecutive timeouts: {consecutive_timeouts}.")


    # Stop the chatbot's internal async loop running in the thread.
    # Use asyncio.run() again just to await the stop() method.
    logging.info("Stopping Chatbot after sync use.")
    try:
        # Add a timeout for the stop operation itself to prevent hangs.
        # Use asyncio.wait_for for the stop operation.
        asyncio.run(asyncio.wait_for(chatbot.stop(), timeout=10)) # 10 second timeout for stop
        logging.info("Chatbot stopped after sync use.")
    except asyncio.TimeoutError:
        logging.error("Chatbot stop operation timed out!")
    except Exception as e:
        logging.exception(f"Error during chatbot stop for sync use: {e}")


if __name__ == "__main__":
    # Run the async example first
    asyncio.run(run_async_example())

    print("\n" + "="*60 + "\n")

    # Run the synchronous example
    run_sync_example()

    print("\n" + "="*60 + "\n")
    logging.info("Both examples finished.")