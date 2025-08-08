"""
Module for the CLI to experiment with chatting using a pre-configured chatbot.
"""
import argparse

from pyperclip import init_windows_clipboard

from llms_wrapper.config import read_config_file as read_config
from fasthtml.common import *
from loguru import logger
from llms_wrapper.llms import LLMS
from llms_wrapper.utils import load_prompts
from llms_wrapper.chatbot import SimpleSerialChatbot
from llms_wrapper.log import configure_logging

def get_args():
    """
    Get the command line arguments
    """
    parser = argparse.ArgumentParser(description='llms_wrapper simple chatbot version')
    parser.add_argument('--port', '-p', type=int, default=50001, help='Port to run the web server on (default: 50001)', required=False)
    parser.add_argument('--config', '-c', type=str, help='Configuration file, should contain LLM and prompt config', required=True)
    parser.add_argument("--llm", '-l', type=str, help="Alias of LLM to use for chatbot (or use first found in config)", required=False)
    parser.add_argument("--promptfile", type=str, help="Prompt file containing prompts, overrides config (or use prompt section in config)", required=False)
    parser.add_argument("--pids", nargs="*", help="Prompt ids (pids) to use as initial messages for the chatbot")
    parser.add_argument("--max_messages", type=int, default=20,
                        help="Max number of messages to keep in the chat history (default: 20)", required=False)
    parser.add_argument("--clear_page", action="store_true",
                        help="Clear the page too when history is cleared (default: keep, add info message)", required=False)
    parser.add_argument('--verbose', action="store_true",
                        help='Be more verbose', required=False)
    parser.add_argument("--debug", action="store_true", help="Debug mode, overrides loglevel", required=False)
    args = parser.parse_args()
    configure_logging(level="DEBUG" if args.debug else "INFO")
    config = {}
    config.update(vars(args))
    # debug implies verbose
    if args.debug:
        config["verbose"] = True
    else:
        config["verbose"] = False
    fconfig = read_config(args.config)
    # we want the argument values to override the config file values so we only update the values that do not
    # exist or are None in the config file
    for k, v in fconfig.items():
        if k not in config or config[k] is None:
            config[k] = v

    # if a promptfile is specified, load the prompts from the file and update the config
    if args.promptfile is not None:
        prompts = load_prompts(args.promptfile, as_messages=True)    # the prompt gets returned as messages!
        if "prompts" not in config:
             config["prompts"] = {}
        elif not isinstance(config["prompts"], dict):
            logger.error("Prompts in config file are not a dictionary mapping a prompt id to a prompt")
        for pid, prompt in prompts.items():
            if pid in config["prompts"]:
                logger.warning(f"Prompt {pid} already exists in config file, overriding from the prompt file!")
            config["prompts"][pid] = prompt
    if args.pids: # if pids are specified, collect them for later
        init_messages = []
        for pid in args.pids:
            if pid in config["prompts"]:
                init_messages += config["prompts"][pid]
            else:
                raise ValueError(f"PID {pid} not found in promptfile")
        logger.info(f"Loaded initial messages: {init_messages}")
        config["init_messages"] = init_messages

    # set the config parameter use_llm to the llm to use
    if not "llms" in config or not isinstance(config["llms"], list) or len(config["llms"]) == 0:
        logger.error("No LLMs defined in config file")
        sys.exit(1)
    if args.llm is not None:
        if args.llm not in [llm["alias"] for llm in config["llms"]]:
            logger.error(f"LLM {args.llm} not found in config file")
            sys.exit(1)
        config["use_llm"] = args.llm
    else:
        # if no llm is specified, use the first one in the config file
        config["use_llm"] = config["llms"][0]["alias"]
    return config

config = get_args()
llms = LLMS(config)
llm_aliases = [l["alias"] for l in config["llms"]]
llm_used = config["use_llm"]
logger.info(f"Initial LLM: {llm_used}")
llm = llms[llm_used]

chatbot = SimpleSerialChatbot(
    llm,
    config=config,
    initial_message=config.get("init_messages"),
    message_template=None,
    max_messages = config["max_messages"],  # Max number of messages to keep in the chat history)
)

# Set up the app, including daisyui and tailwind for the chat component
hdrs = (
    picolink,
    Script(src="https://cdn.tailwindcss.com"),
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"),
    Script(src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"),
    # add a style for the success bubble which is too light, make it a darker green
    Style("""
           .chat-bubble {
               --pico-color: white !important;
               --pico-background-color: transparent !important;
               color: var(--chat-text-color, white) !important;
      --pico-h1-color: white !important;
      --pico-h2-color: white !important;
      --pico-h3-color: white !important;
      --pico-h4-color: white !important;
      --pico-h5-color: white !important;
      --pico-h6-color: white !important;
      --pico-muted-color: white !important;
      --pico-primary-color: white !important;
      --pico-secondary-color: white !important;
      --pico-contrast-color: white !important;
      --pico-marked-color: white !important;
      --pico-ins-color: white !important;
      --pico-del-color: white !important;
      --pico-blockquote-footer-color: white !important;
      --pico-blockquote-border-color: rgba(255,255,255,0.3) !important;
      --pico-table-border-color: rgba(255,255,255,0.2) !important;
      --pico-code-color: white !important;
      --pico-pre-color: white !important;
      --pico-kbd-color: white !important;
      --pico-mark-color: white !important;
           }
           .chat-bubble * {
               color: inherit !important;
           }
           .chat .chat-bubble.chat-bubble-secondary {
               background-color: #126e13 !important; /* or #228B22 */
               color: white !important; /* Text color for readability */
           }
           .chat .chat-bubble.chat-bubble-primary {
               background-color: #3f3b6b !important; /* or #228B22 */
               color: white !important; /* Text color for readability */
           }
           chat .chat-bubble.chat-bubble-other {
               background-color: blue !important; /* or #228B22 */
               color: white !important; /* Text color for readability */
           }
            .btn.btn-primary.btn-disabled.thinking-red-text {
               color: #EF4444 !important; /* Tailwind's red-500 color, with !important to override disabled styles */
               opacity: 1 !important; /* Ensure text is not made transparent by disabled styles */
            }
           .markdown-content ul,
           .markdown-content ol {
               /* Re-enable default list style types explicitly */
               list-style-type: disc; /* For unordered lists */
               /* list-style-type: decimal; is default for <ol> but let's be explicit */
               list-style-position: outside; /* Crucial: make bullets/numbers appear outside the text block */
               margin-left: 2em; /* Increase indent to make space for bullet/number */
               padding-left: 0; /* Ensure no conflicting padding */
           }

           .markdown-content ol {
               list-style-type: decimal; /* Explicitly set for ordered lists */
           }

           .markdown-content li {
               margin-bottom: 0.2em; /* Space between list items */
               /* Ensure no hidden styling for list items */
               display: list-item; /* This is the default, but good to be explicit if issues persist */
           }

           .markdown-content p {
               margin-top: 0.5em; /* Space between paragraphs */
               margin-bottom: 0.5em;
           }

           .markdown-content strong {
               font-weight: bold;
           }

           .markdown-content em {
               font-style: italic;
           }

           .markdown-content pre {
               padding: 0.75em;
               border-radius: 0.375rem;
               overflow-x: auto;
               font-family: monospace;
               white-space: pre-wrap;
           }

           .markdown-content code {
               padding: 0.1em 0.3em;
               border-radius: 0.25rem;
               font-family: monospace;
           }           
           """)
)
# app = FastHTML(hdrs=hdrs, cls="p-4 max-w-lg mx-auto")
# themes: dark, light, cupcake, bumblebee, emerald, corporate, synthwave, retro, cyberpunk, valentine, halloween, garden, forest, aqua, lofi, pastel, fantasy,
#     wireframe, black, luxury, dracula, cmyk, autumn, business, acid, lemonade, night, coffee, winter, dim, nord, sunset, caramellatte, abyss, silk
app = FastHTML(hdrs=hdrs, cls="p-4 max-w-full max-w-3xl mx-auto", data_theme="business")

# Chat message component (renders a chat bubble)
# kind is one of user, assistant, error
def ChatMessage(msg, kind):
    # other classes: chat-bubble-accent/neutral/info/success/warning/error
    if kind == "error":
        bubble_class = "chat-bubble-error"
    elif kind == "warning":
        bubble_class = "chat-bubble-other"
    elif kind == "user":
        bubble_class = "chat-bubble-primary"
    elif kind == "assistant":
        bubble_class = "chat-bubble-secondary"
    else:
        bubble_class = "chat-bubble-other"  # Default to warning if kind is unknown
    chat_class = "chat-end" if kind == "user" else 'chat-start'
    return Div(cls=f"chat {chat_class}")(
               # No content in the Div directly, it will be filled by JS
               # data-markdown stores the raw Markdown string
               Div(cls=f"chat-bubble {bubble_class} markdown-content", **{'data-markdown': msg}),
               # Hidden(msg, name="messages"), # This Hidden field is for sending messages back to server if needed
           )

# The input field for the user message. Also used to clear the
# input field after sending a message via an OOB swap
def ChatInput():
    return Input(name='msg', id='msg-input', placeholder="Type a message",
                 cls="input input-bordered w-full", hx_swap_oob='true',
                 oninput="checkMessageInput()", hx_on="keyup: if (event.key === 'Enter') { sendMessage(); }",
                 )

button_script = Script("""
    let appState = "ready"; 
    function setAppState(state) {
        appState = state;
        updateSendButton();
    }
    
    function checkMessageInput() {
        const input = document.getElementById('msg-input');
        const sendButton = document.getElementById('send-button');
        if (input && sendButton) {
            updateSendButton();
        }
    }
    
    function updateSendButton() {
        const input = document.getElementById('msg-input');
        const sendButton = document.getElementById('send-button');
        if (!input || !sendButton) return;
        const isEmpty = input.value.trim() === '';
        if (appState == 'busy') {
            sendButton.disabled = true; // Disable button if app is busy
            sendButton.textContent = "Thinking"; // Change button text
            sendButton.classList.add("btn-disabled"); 
            sendButton.classList.add("thinking-red-text"); // Add red text class
        } else {
            sendButton.classList.remove("thinking-red-text"); // Remove red text class
            sendButton.textContent = "Send"; // Reset button text
            sendButton.disabled = isEmpty; // Enable/disable based on input
            if (isEmpty) {
                sendButton.classList.add("btn-disabled"); // Add disabled class
            } else {
                sendButton.classList.remove("btn-disabled"); // Remove disabled class
            }
        }
    }
    
    function updateClearButton() {
        const clearButton = document.getElementById('clear-button');
        if (clearButton) {
            clearButton.disabled = appState === 'busy';
        }
    }

    // Call on page load to set initial state
    document.addEventListener('htmx:load', checkMessageInput); // Use htmx:load for initial and subsequent loads
 
    // Re-check after HTMX swaps (e.g., after sending a message and clearing the input)
    document.addEventListener('htmx:afterSwap', function(event) {
        // If the swap involved the input field being replaced (OOB swap)
        // or any content that might affect the button state
        updateClearButton();
        checkMessageInput();
    });
    
    function sendMessage() {
        const input = document.getElementById('msg-input');
        if (input.value.trim() !== '' && appState === 'ready') {
            setAppState('busy'); // Set app state to busy
            htmx.trigger(document.getElementById("chat-form"), "submit"); // Trigger form submission
        }
    }
    
    document.addEventListener('htmx:beforeRequest', function(event) {
        if (event.target.id === "chat-form") {
            setAppState('busy');
        }
        updateClearButton(); // Update clear button state
    });
    
    setAppState('ready'); // Initialize app state to ready
    
""")

markdown_render_script = Script("""
    document.addEventListener('htmx:afterSwap', function(event) {
        const markdownElements = event.detail.target.querySelectorAll('.markdown-content');
        markdownElements.forEach(function(element) {
            const rawMarkdown = element.dataset.markdown;
            if (rawMarkdown) {
                console.log("Raw Markdown:", rawMarkdown); // Log the input to marked.js
                const htmlOutput = marked.parse(rawMarkdown);
                console.log("Marked.js HTML Output:", htmlOutput); // Log the HTML marked.js generates
                element.innerHTML = htmlOutput;
            }
        });
    });
""")


def SelectBox(options, selected_option=""):
    # Create Option elements for the select box
    select_options = [
        Option(opt, value=opt, selected=(opt == selected_option))
        for opt in options
    ]
    return Select(
        *select_options,
        name="selected_option",
        id="select-option",
        cls="select select-bordered flex-grow-0 max-w-[10rem]",
        autocomplete="off",
    )


def InfoMessageDiv(message="", kind="info"):
    alert_class = ""
    if kind == "info":
        alert_class = "text-blue-500" # Tailwind class for blue text
    elif kind == "success":
        alert_class = "text-green-500" # Tailwind class for green text
    elif kind == "warning":
        alert_class = "text-yellow-500" # Tailwind class for yellow text
    elif kind == "error":
        alert_class = "text-red-500" # Tailwind class for red text

    return P(
        message,
        id="info-message", # Must match the ID in index()
        hx_swap_oob="true", # This is crucial for OOB swap
        cls=alert_class # Apply styling here
    )

# The main screen
@app.get
def index():
    page = Form(hx_post=send, hx_target="#chatlist", hx_swap="beforeend", id="chat-form",
                autocomplete="off", # Disable browser autocomplete, so values from last session should not be used
                )(
           Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
               Div(cls="flex space-x-2 mt-2")(
                   Group(
                       SelectBox(llm_aliases, selected_option=llm_used),
                       ChatInput(),
                       Button("Send", cls="btn btn-primary", id="send-button", disabled=True, onclick="sendMessage()"),
                       Button("Clear", cls="btn btn-secondary", id="clear-button",
                              hx_post="/clear_chat",
                              # hx_target="#chatlist",
                              # hx_swap="innerHTML",
                              hx_include="#msg-input, #select-option",
                              hx_trigger="click"),
                   )
               ),
               Div(cls="flex space-x-2 mt-2", id="info-message")(
                   P("Welcome to the Chatbot! Enter a question or \"?help\" for special commands.", id="info-message", cls="text-blue-500"),
               )
           )
    return Titled('SimpleChatbot', page), markdown_render_script, button_script

@app.post("/clear_chat")
def clear_chat_history(msg:str = "", selected_option:str = ""): # Parameters to receive from hx_include
    chatbot.clear_history()
    # Return empty Div for chatlist to clear it, and an empty input for OOB swap
    logger.info("LLM chat history cleared")
    if config["clear_page"]:
        logger.info("Clearhing web page")
        # If clear_page is set, return an empty Div to clear the chat list
        return (Div(id="chatlist", cls="chat-box"),  # This clears the chat list, keeping its class
                ChatInput(), InfoMessageDiv("Chat history cleared!", kind="info"))
    else:
        # Do not clear the chat list, just add an info that the LLM chat history was cleared
        logger.info("Chat history cleared, but keeping the chat page")
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatMessage("LLM chat history has been reset!", "other"),  # The chatbot's response
            ChatInput(), InfoMessageDiv("", "info"))  # And clear the input field via an OOB swap


# Handle the form submission
@app.post
def send(msg:str, selected_option:str=""):
    msg = msg.strip()  # Strip whitespace from the message
    logger.info(f"Received message: {msg}, selected option: {selected_option}")
    info_txt = ""
    if selected_option:
        # get the current LLM used
        active_llm = chatbot.llm["alias"]
        if selected_option != active_llm:
            llm = llms[selected_option]
            chatbot.set_llm(llm)
            info_txt =  f"LLM changed from {active_llm} to {selected_option}. "
            logger.warning(info_txt+"<br>")   # TODO: need to make the div accept raw HTML

    info_empty = InfoMessageDiv("", kind="success")
    if not chatbot:
        errmsg = "Chatbot not initialized"
        logger.error(errmsg)
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatMessage(errmsg, "error"), # The chatbot's response
            ChatInput(), info_empty)
    if not msg or msg.strip() == "":
        info_txt += "No message provided, ignored"
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatInput(), InfoMessageDiv(info_txt, kind="error"))
    msg = msg.strip()  # Ensure no leading/trailing whitespace
    if msg == "help" or msg == "?" or msg == "?help":
        chatmsg = """
        Available commands:
        - `?help`: Show this help message
        - `?clear`: Clear the chat history
        - `?history`: Show the history of all user requests and chatbot responses
        """
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatMessage(chatmsg, "assistant"), # The chatbot's response
            ChatInput(), InfoMessageDiv(info_txt, kind="info"))
    elif msg == "?clear":
        chatbot.clear_history()
        # Return empty Div for chatlist to clear it, and an empty input for OOB swap
        logger.info("Chat history cleared")
        if config["clear_page"]:
            logger.info("Clearing web page")
            # If clear_page is set, return an empty Div to clear the chat list
            return (Script("setAppState('ready');"),
                    Div(id="chatlist", cls="chat-box"),  # This clears the chat list, keeping its class
                    ChatInput(), InfoMessageDiv("Chat history cleared!", kind="info"))
        else:
            # Do not clear the chat list, just add an info that the chat history was cleared
            logger.info("Chat history cleared, but keeping the chat page")
        return ( Script("setAppState('ready');"),
                 ChatMessage("LLM chat history cleared", "other"),
                 ChatInput(), InfoMessageDiv("Chat history cleared!", kind="info"))  # Updates the info message
    elif msg == "?history":
        # Show the history of all user requests
        history = chatbot.history
        ret = [Script("setAppState('ready');")]
        for req,resp in history:
            ret.append(ChatMessage(req, "user"))
            ret.append(ChatMessage(resp, "assistant"))
        ret.append(ChatInput())
        info_txt += "Showing chat history."
        info_div = InfoMessageDiv(info_txt, kind="info")
        ret.append(info_div)
        return ret
    elif msg.startswith("?"):
        # Handle other commands starting with '?'
        errmsg = f"Unknown command: {msg}. Type ?help for available commands."
        logger.error(errmsg)
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatMessage(msg, "user"),    # The user's message
            ChatMessage(errmsg, "error"), # The chatbot's response
            ChatInput(), InfoMessageDiv(info_txt, "error"))
    try:
        logger.info(f"Processing user message: {msg}")
        response = chatbot.reply(msg)
        error = response["error"]
        answer = response["answer"]
        is_ok = response["is_ok"]
    except Exception as e:
        error = str(e)
        answer = None
        is_ok = False
        logger.error(f"Chatbot error: {error}")
    if not is_ok:
        errmsg = f"PROCESSING ERROR: {error}"
        info_txt += f"{errmsg}"
        return (
            Script("setAppState('ready');"),  # Reset the app state to ready
            ChatMessage(msg, "user"),    # The user's message
            ChatMessage(errmsg, "error"), # The chatbot's response
            ChatInput(), InfoMessageDiv(info_txt, "error")) # And clear the input field via an OOB swap
    else:
        logger.info(f"Chatbot response:\n=================================\n{answer}\n=================================")
        n_prompt_tokens = response["n_prompt_tokens"]
        n_completion_tokens = response["n_completion_tokens"]
        cost = response["cost"]
        total_cost = chatbot.llm.cost()
        all_cost = llms.cost()
        logger.info(f"Tokens sent: {n_prompt_tokens}, received: {n_completion_tokens}")
        logger.info(f"Cost: {cost:.4f} USD, total LLM cost so far: {total_cost:.4f} USD")
        logger.info(f"Total cost all LLMS so far: {total_cost:.4f} USD")
        info_msg = InfoMessageDiv(
            info_txt+
            f"Tokens sent: {n_prompt_tokens}, received: {n_completion_tokens}, "
            f"Cost: {cost:.4f} USD, total cost so far: {total_cost:.4f} USD, "
            f"Total cost all LLMS so far: {all_cost:.4f} USD",
            kind="success"
        )
        return (
            Script("setAppState('ready');"), # Reset the app state to ready
            ChatMessage(msg, "user"),    # The user's message
            ChatMessage(answer, "assistant"), # The chatbot's response
            ChatInput(), info_msg) # And clear the input field via an OOB swap


def main():
    serve(#app,
          appname="llms_wrapper.llms_wrapper_webchat",
          host="127.0.0.1",
          port=config["port"],
          reload=False,
    )

if __name__ == "__main__":
    main()
