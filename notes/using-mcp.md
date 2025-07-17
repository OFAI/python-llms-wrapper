# Notes on using MCP


## MCP 

See https://github.com/modelcontextprotocol/python-sdk


Overall approach:

```
import os
from litellm import experimental_mcp_client
from litellm import completion
from mcp import ClientSession, StdioServerParameters
from mcp.client.http import http_client
from mcp.client.stdio import stdio_client

sparms = StdioServerParameters(command="npx", args=["@wonderwhy-er/desktop-commander@latest"])

async with stdio_client(sparms) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        # messages = ...
        # response = await acompletion(model=, messages=, tools=mcp_tools, tool_choice="auto")
        # message = response.choices[0].message
        # if message.tool_calls:
        #     for tool_call in message.tool_calls:
        #         function_name = tool_call.function.name
        #         function_args = json.loads(tool_call.function.arguments)
        #         tool_result = await experimental_mcp_client.call_openai_tool(session=session, tool_call=tool_call)
        #         messages.append(dict(tool_call_id=tool_call.id, role="tool", content=str(tool_result.content)))
        #     final_response = await acompletion(model=, messages=messages, tools=mcp_tools) 

```


In order to avoid moving all code into async functions, refactor all MCP async processing into a class:

```
# mcp_handler.py
import asyncio
import threading
import concurrent.futures # For Future.result() timeout

from typing import Dict, Any, List, Tuple, Set

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.http import http_client
from litellm import experimental_mcp_client

class McpToolHandler:
    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        self._server_configs = server_configs
        self._mcp_sessions: Dict[str, ClientSession] = {}
        self._mcp_tool_name_to_session_id: Dict[str, str] = {}
        self._session_contexts: List[Any] = [] # To store context managers for cleanup
        self._all_mcp_tools: List[Dict[str, Any]] = [] # Stored loaded tools for LLM

        # Threading for the asyncio event loop
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._is_initialized = False

    def _start_loop_thread(self):
        """Starts the asyncio event loop in a new thread."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, args=(self._loop,), daemon=True)
        self._loop_thread.start()
        print("MCP asyncio event loop started in a background thread.")

    def _run_loop(self, loop: asyncio.AbstractEventLoop):
        """Target function for the asyncio thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _run_async_method_sync(self, coro):
        """Helper to run an async coroutine from a synchronous context."""
        if not self._is_initialized or not self._loop.is_running():
            raise RuntimeError("McpToolHandler is not initialized or loop is not running.")
        
        # Submit the coroutine to the event loop in the other thread
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        # Block the current thread until the coroutine in the other thread completes
        # Add a timeout to prevent indefinite hangs
        try:
            return future.result(timeout=60) # Increased timeout for potential long-running MCP tools
        except concurrent.futures.TimeoutError:
            print(f"Error: MCP tool execution timed out after 60 seconds for coroutine {coro}")
            future.cancel() # Attempt to cancel the task in the other thread
            raise
        except Exception as e:
            print(f"Error running async method sync: {e}")
            raise


    def initialize(self) -> List[Dict[str, Any]]:
        """
        Synchronously initializes all MCP client sessions and loads their tools.
        Returns a list of all loaded MCP tool schemas in OpenAI format.
        """
        if self._is_initialized:
            print("McpToolHandler already initialized.")
            return self._all_mcp_tools

        print("Initializing MCP Tool Handler (synchronously)...")
        self._start_loop_thread() # Start the background async loop

        # Submit the actual async initialization logic to the new thread
        # This part still needs to be an async coroutine, but it's run via _run_async_method_sync
        async def _async_init_logic():
            for server_id, config in self._server_configs.items():
                session = None
                context_manager = None

                if config["type"] == "stdio":
                    server_params = StdioServerParameters(command=config["command"], args=config["args"])
                    context_manager = stdio_client(server_params)
                    read, write = await context_manager.__aenter__()
                    session = ClientSession(read, write)
                elif config["type"] == "http":
                    context_manager = http_client(config["url"])
                    session = await context_manager.__aenter__()
                else:
                    print(f"Warning: Unknown MCP server type for {server_id}: {config['type']}")
                    continue

                self._session_contexts.append(context_manager)
                self._mcp_sessions[server_id] = session
                await session.initialize()
                print(f"Initialized MCP session for {server_id}")

                server_tools = await experimental_mcp_client.load_mcp_tools(
                    session=session,
                    format="openai"
                )
                print(f"Loaded {len(server_tools)} tools from {server_id}")

                for tool in server_tools:
                    tool_name = tool['function']['name']
                    if tool_name in self._mcp_tool_name_to_session_id:
                        raise ValueError(f"Duplicate tool name found across MCP servers: '{tool_name}'. Please ensure all MCP tool names are unique.")
                    self._mcp_tool_name_to_session_id[tool_name] = server_id
                    self._all_mcp_tools.append(tool)
            
            print(f"MCP Tool Handler initialized with {len(self._all_mcp_tools)} tools.")
            return self._all_mcp_tools
        
        # Run the async initialization logic in the background loop, waiting for its completion
        self._all_mcp_tools = self._run_async_method_sync(_async_init_logic())
        self._is_initialized = True
        return self._all_mcp_tools

    def execute_mcp_tool(self, tool_call: Any) -> str:
        """
        Synchronously executes an MCP tool call by submitting it to the
        background asyncio loop.
        """
        function_name = tool_call.function.name
        session_id = self._mcp_tool_name_to_session_id.get(function_name)

        if not session_id:
            return f"Error: MCP tool '{function_name}' not found or not associated with any session."

        target_session = self._mcp_sessions[session_id]

        async def _async_execute_tool():
            print(f"  -> Executing MCP tool '{function_name}' via session '{session_id}' (async background)")
            try:
                tool_result = await experimental_mcp_client.call_openai_tool(
                    session=target_session,
                    tool_call=tool_call
                )
                return str(tool_result.content)
            except Exception as e:
                return f"Error executing MCP tool '{function_name}' from '{session_id}': {e}"
        
        return self._run_async_method_sync(_async_execute_tool())


    def get_mcp_tool_names(self) -> Set[str]:
        """Returns a set of all MCP tool names managed by this handler."""
        return set(self._mcp_tool_name_to_session_id.keys())

    def close(self):
        """Synchronously closes all active MCP client sessions and stops the background loop."""
        print("Closing MCP Tool Handler (synchronously)...")
        if not self._is_initialized:
            print("McpToolHandler not initialized, nothing to close.")
            return

        # Submit the async cleanup logic to the background loop
        async def _async_cleanup_logic():
            for context_manager in self._session_contexts:
                await context_manager.__aexit__(None, None, None)
            self._mcp_sessions.clear()
            self._mcp_tool_name_to_session_id.clear()
            self._session_contexts.clear()
            print("All MCP sessions closed in background loop.")
        
        # Run cleanup, blocking until it's done
        self._run_async_method_sync(_async_cleanup_logic())

        # Stop the background event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5) # Wait for thread to finish
            if self._loop_thread.is_alive():
                print("Warning: Background asyncio thread did not stop gracefully.")
        
        self._is_initialized = False
        print("MCP Tool Handler closed.")
```

and then use it like this:

```
MCP_SERVER_CONFIGS = {
    "Server1": dict(type="stdio", command="npx", args=["@wonderwhy-er/desktop-commander@latest"]),
    "Server2": dict(type="http", url="http://localhost:8801"),
}
mcp_handler = McpToolHandler(MCP_SERVER_CONFIGS)
mcp_tools_for_llm = mcp_handler.initialize()
mcp_tool_names = mcp_handler.get_mcp_tool_names()
# TODO: make sure the local tools do not have names identical to any of the MCP tools
all_tools = mcp_tools_for_llm + python_tools

# ... create messages, invoke llm
# .. then when going through any tool calls:
if tool_call.function.name in mcp_tool_names:
    tool_result = mcp_handler.execute_mcp_tool(tool_call)
else:
    # run local tool
# extend message etc.
```



## Package mcp_use 

See https://github.com/mcp-use/mcp-use

## FastMCP 

See https://github.com/jlowin/fastmcp

E.g. using the desktop commander:

```
config = dict(mcpServers=dict(dc=dict(command="npx", args=["@wonderwhy-er/desktop-commander@latest"])))

async def test():
    # Connect via stdio to a local script
    async with Client(config) as client:
        tools = await client.list_tools()
        result = await client.call_tool("get_config")
    return tools, result
```

The var `tools` contains a list of Tool instances with attrs name, title, description a.o.


