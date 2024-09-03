from typing import List
from fasthtml.common import *
from mlx_lm import load, stream_generate
from functools import partial
from huggingface_hub import HfApi, snapshot_download, scan_cache_dir, CacheNotFound
import os
from starlette.requests import Request
from fasthtml.components import Zero_md
import threading

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
zeromd_script = Script(type="module", src="https://cdn.jsdelivr.net/npm/zero-md@3?register")
custom_style = Style("""
.chat-bubble-secondary {
    background-color: #8f1e53 !important;
    color: white !important;
}
input[type="radio"] {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    border: 2px solid #555;
    outline: none;
    margin-right: 5px;
    flex-shrink: 0;
}
input[type="radio"]:checked {
    background-color: #555;
}
.model-name {
    display: flex;
    flex-direction: column;
    line-height: 1.2;
    word-break: break-word;
}
.model-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 0.5rem;
}
.resizable-layout {
    display: flex;
    height: 100vh;
    width: 100%;
}
.sidebar {
    flex: 0 0 250px;
    transition: all 0.3s ease;
    overflow: hidden;
    position: relative;
    max-width: 250px;
    display: flex;
    flex-direction: column;
    height: calc(100vh - 50px); /* Adjust height based on button height */
    overflow-y: auto; /* Allow scrolling if content overflows */
}
.sidebar.hidden {
    flex: 0 0 0;
    max-width: 0;
}
.sidebar-content {
    padding: 1rem;
    width: 100%;
    overflow-y: auto;  // Make the content scrollable
    flex-grow: 1;  // Allow the content to grow and fill the sidebar
}
.sidebar-right {
    flex: 0 0 350px;  /* Wider right sidebar */
    max-width: 350px;  /* Matching max-width */
}
.toggle-sidebar {
    position: fixed;
    top: 50%;
    transform: translateY(-50%);
    background: #007bff; //
    border: none;
    cursor: pointer;
    z-index: 10;
    font-size: 14px;
    transition: all 0.3s ease;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    height: 100px;
    width: 8px;  // Always start slim
    border-radius: 0 3px 3px 0;
}
                     
.toggle-sidebar:hover {
    width: 30px;
}
#toggle-left-sidebar {
    left: 0;
}
#toggle-right-sidebar {
    right: 0;

}
.toggle-sidebar span {
    opacity: 0;
    transition: opacity 0.3s ease;
}
.toggle-sidebar:hover span {
    opacity: 1;
}
.main-content {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
    min-width: 0;
    transition: margin 0.3s ease;
    padding: 0 2.5%;  // Add padding to the main content
}
.chat-container {
    width: 100%;
    max-width: 95%;  // Set maximum width to 95%
    margin: 0 auto;  // Center the chat container
}
.chat-box {
    height: 73vh;
    overflow-y: auto;
    border: 1px solid #e0e0e0;  // Add a light border
    border-radius: 8px;  // Rounded corners
    padding: 1rem;
}
.chat-input-container {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
}
.chat-input {
    flex-grow: 1;
}
#theme-toggle {
    margin: 1rem; /* Add margin for spacing */
    padding: 0.5rem 1rem; /* Add padding for better appearance */
    background-color: #007bff; /* Button background color */
    color: white; /* Button text color */
    border: none; /* Remove default border */
    border-radius: 4px; /* Rounded corners */
    cursor: pointer; /* Pointer cursor on hover */
}
#theme-toggle:hover {
    background-color: #0056b3; /* Darker shade on hover */
}
                     
.button-section { 
    padding: 1rem; /* Add padding for spacing */ 
    background-color: #f9f9f9; /* Light background for contrast */ 
    padding: 0.5rem;
}

/* Light theme styles */
[data-theme="light"] .button-section {
    background-color: #f9f9f9; /* Light background for light mode */
    color: black; /* Text color for light mode */
}

/* Dark theme styles */
[data-theme="dark"] .button-section {
    background-color: #333; /* Dark background for dark mode */
    color: white; /* Text color for dark mode */
}                     

.small-button, .btn {
    background-color: #007bff; /* Unified button background color */
    color: white; /* Button text color */
    border: none; /* Remove default border */
    border-radius: 4px; /* Rounded corners */
    cursor: pointer; /* Pointer cursor on hover */
}

.small-button:hover, .btn:hover {
    background-color: #0056b3; /* Darker shade on hover */
}

.flex {
    display: flex; /* Use flexbox for layout */
    align-items: center; /* Center items vertically */
}

.ml-2 {
    margin-left: 0.5rem; /* Add left margin to the heading */
}
""")

resize_script = Script("""
function initToggleableSidebars() {
    const leftSidebar = document.getElementById('left-sidebar');
    const rightSidebar = document.getElementById('right-sidebar');
    const mainContent = document.querySelector('.main-content');
    const toggleLeftBtn = document.getElementById('toggle-left-sidebar');
    const toggleRightBtn = document.getElementById('toggle-right-sidebar');

    function toggleSidebar(sidebar, button, isLeft) {
        sidebar.classList.toggle('hidden');
        if (isLeft) {
            mainContent.style.marginLeft = sidebar.classList.contains('hidden') ? '0' : '';
            button.querySelector('span').textContent = sidebar.classList.contains('hidden') ? 'Open Left' : 'Close Left';
        } else {
            mainContent.style.marginRight = sidebar.classList.contains('hidden') ? '0' : '';
            button.querySelector('span').textContent = sidebar.classList.contains('hidden') ? 'Open Right' : 'Close Right';
        }
    }

    toggleLeftBtn.addEventListener('click', () => toggleSidebar(leftSidebar, toggleLeftBtn, true));
    toggleRightBtn.addEventListener('click', () => toggleSidebar(rightSidebar, toggleRightBtn, false));
}

document.addEventListener('DOMContentLoaded', initToggleableSidebars);
""")
hyperscript = Script(src="https://unpkg.com/hyperscript.org@0.9.11")
app = FastHTML(hdrs=(tlink, dlink, picolink, zeromd_script, custom_style, resize_script, hyperscript), htmlkw={"data-theme": "light"})

# Global variables for model and tokenizer
model = None
tokenizer = None
system_message = "You are a helpful AI assistant."
messages = [{"role": "system", "content": system_message}]
is_generating = False  # New flag to track generation status
is_loading_model = False  # New flag to track model loading status
temperature = 0.7
max_tokens = 512

def scan() -> List[str]:
    """Scans the Hugging Face cache directory and returns a list of Models."""
    try:
        hf_cache_info = scan_cache_dir()
        models = [model.repo_id for model in hf_cache_info.repos]
        # Sort alphabetically
        models.sort()
        return [model for model in models if model.startswith("mlx-community/") and "whisper" not in model]
    except CacheNotFound:
        return []

# Function to load a model
def load_model(model_dir):
    global model, tokenizer
    model, tokenizer = load(model_dir)
    model_name = os.path.basename(model_dir)
    return f"Using {model_name}"

# Function to search for MLX models
def search_mlx_models(query):
    api = HfApi()
    models = api.list_models(
        author="mlx-community",
        search=query
    )
    return [model.id for model in models]

# Function to download a model
@app.post("/download_model/{user_id}/{model_id}")
async def download_model_route(user_id: str, model_id: str):
    try:
        directory_path = os.path.dirname(os.path.abspath(__file__))
        local_model_dir = os.path.join(directory_path, "models", "download", model_id.split("/")[-1])
        
        status_id = f"status-{user_id}-{model_id}"
        
        # Start the download in a separate thread
        thread = threading.Thread(target=download_model_thread, args=(user_id, model_id, local_model_dir, status_id))
        thread.start()
        

    except Exception as e:
        return Div(f"Error starting model download: {str(e)}", id="download_status")

def download_model_thread(user_id, model_id, local_model_dir, status_id):
    try:
        snapshot_download(
            repo_id=f"{user_id}/{model_id}",
            resume_download=True
        )

    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        # You might want to update an error status here if needed
# Function to render markdown
def render_local_md(md, css=''):
    css_template = Template(Style(css), data_append=True)
    return Zero_md(css_template, Script(md, type="text/markdown"))

# CSS to fix styling issues
css = '.markdown-body {background-color: unset !important; color: unset !important;}'
_render_local_md = partial(render_local_md, css=css)

# Chat message component
def ChatMessage(msg_idx):
    msg = messages[msg_idx]
    if msg['role'] == 'system':
        return ''  # Return an empty string for system messages
    text = "..." if msg['content'] == "" else msg['content']
    bubble_class = "chat-bubble-primary" if msg['role']=='user' else 'chat-bubble-secondary'
    chat_class = "chat-end" if msg['role']=='user' else 'chat-start'
    generating = 'generating' in messages[msg_idx] and messages[msg_idx]['generating']
    stream_args = {
        "hx-trigger": "every 0.5s",
        "hx-swap": "outerHTML",
        "hx-get": f"/chat_message/{msg_idx}"
    } if generating else {}
    
    # Render the content as markdown
    rendered_content = _render_local_md(text)
    
    return Div(Div(msg['role'], cls="chat-header"),
               Div(rendered_content, 
                   cls=f"chat-bubble {bubble_class}",
                   id=f"message-content-{msg_idx}"),
               cls=f"chat {chat_class} max-w-full", 
               id=f"chat-message-{msg_idx}",
               **stream_args)

# Route that gets polled while streaming
@app.get("/chat_message/{msg_idx}")
def get_chat_message(msg_idx:int):
    if msg_idx >= len(messages):
        return ""
    return ChatMessage(msg_idx)

# Add this new function to render the system message input
def SystemMessageInput():
    return Div(
        Label("System Message:", fr="system_message", cls="block mb-2"),
        Textarea(system_message, name="system_message", id="system_message",
                 cls="textarea textarea-bordered w-full",
                 rows="3",
                 hx_post="/update_system_message",
                 hx_trigger="change",
                 hx_target="#system_message_status"),
        Div(id="system_message_status", cls="mt-2 text-sm"),
        cls="mb-4"
    )

# Add this function to toggle the theme
@app.get("/")
def get():
    return Title('MLX Chatbot'), Div(
        Div(
            Div(
                Button("Dark", id="theme-toggle", cls="small-button"),  # Button with initial text "Dark"
                H1("MLX Chat", cls="ml-5"),  # Heading next to the button
                cls="flex items-center"  # Flex container for horizontal alignment
            ),
            cls="button-section border-b border-gray-300 pb-4 mb-4"  # New container with bottom border
        ),
        Script("""
            document.getElementById('theme-toggle').onclick = function() {
                const button = document.getElementById('theme-toggle');
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', newTheme);
                button.textContent = newTheme === 'light' ? 'Dark' : 'Light';  // Update button text
            };
        """),
        Div(
            Button(Span("Close Left"), id="toggle-left-sidebar", cls="toggle-sidebar"),
            Button(Span("Close Right"), id="toggle-right-sidebar", cls="toggle-sidebar"),
            Div(
                Div(
                    H2("Model Search"),
                    Form(
                        Input(type="text", name="model_query", placeholder="Search for MLX models"),
                        Button("Search", cls="btn btn-primary"),
                        Button("Clear", cls="btn btn-secondary ml-2",
                               hx_post="/clear_search",
                               hx_target="#model_list"),
                        hx_post="/search_models", hx_target="#model_list"
                    ),
                    Div(id="model_list"),
                    Div(id="download_status", cls="mt-4"),  # New download status div
                    cls="sidebar-content"
                ),
                id="left-sidebar",
                cls="sidebar"
            ),
            Div(
                Div(  # New container for chat elements
                    Div(id="chatlist", cls="chat-box"),
                    Form(
                        Textarea(name='msg', id='msg-input',
                                 placeholder="Type a message",
                                 cls="textarea textarea-bordered w-full chat-input",
                                 rows="1",
                                 hx_post="/",
                                 hx_trigger="keydown[key=='Enter' && !shiftKey] from:#msg-input",
                                 hx_target="#chatlist",
                                 hx_swap="innerHTML"),
                        Button("Send", cls="btn btn-primary"),
                        hx_post="/", hx_target="#chatlist", hx_swap="innerHTML",
                        cls="chat-input-container"
                    ),
                    Div(
                        Button("Clear Messages", cls="btn btn-secondary",
                               hx_post="/clear_messages",
                               hx_target="#chatlist"),
                        Button("Stop Generation", cls="btn btn-warning ml-2",
                               hx_post="/stop_generation",
                               hx_swap="none"),
                        Div(id="active_model_status", cls="mt-2 text-sm text-600"),
                        cls="mt-2 mb-5"
                    ),
                    cls="chat-container"
                ),
                cls="main-content"
            ),
            Div(
                Div(
                    SystemMessageInput(),  # Add this line
                    Div(
                        Label("Temperature:", fr="temp", cls="block"),
                        Div(
                            Input(type="range", id="temp", name="temp", 
                                  min="0", max="1", step="0.1", value=str(temperature),
                                  cls="w-full",
                                  hx_post="/update_settings",
                                  hx_trigger="change",
                                  hx_target="#settings_status"),
                            Span(f"{temperature}", id="temperature_value", cls="ml-2"),
                            _="on input from #temp set #temperature_value.innerHTML to event.target.value"
                        ),
                        cls="mb-4"
                    ),
                    Div(
                        Label("Max Tokens:", fr="tokens", cls="block"),
                        Input(type="number", id="tokens", name="tokens", 
                              min="1", max="2048", value=str(max_tokens),
                              cls="w-full",
                              hx_post="/update_settings",
                              hx_trigger="change",
                              hx_target="#settings_status"),
                        cls="mb-4"
                    ),
                    Label("Model Selection"),
                    Div(id="downloaded_models", 
                        hx_get="/list_downloaded_models", 
                        hx_trigger="load, refreshModels from:body"), 
                    Button("Refresh Models", cls="btn btn-secondary mt-2",
                        hx_get="/list_downloaded_models",
                        hx_target="#downloaded_models"),
                    cls="sidebar-content"
                ),
                id="right-sidebar",
                cls="sidebar"
            ),
            cls="resizable-layout"
        ),
        cls="max-w-full mx-auto"
    )

# Add a route to list downloaded models
@app.get("/list_downloaded_models")
def list_downloaded_models():
    cached_models = [model for model in scan() if model.startswith("mlx-community/") and "whisper" not in model]
    model_name = [m.split("/")[1] for m in cached_models]
    cached_models = list(zip(cached_models, model_name))
    return Div(
        *[Div(
            Div(
                Input(type="radio", name="model_select", value=model, id=f"model-{model}",
                      hx_post="/load_model",
                      hx_target="#active_model_status",
                      hx_trigger="change"),
                Label(model_name, fr=f"model-{model}", cls="ml-2 text-sm"),
                Button("Delete", 
                   cls="btn btn-error btn-xs mt-1",
                   hx_delete=f"/delete_model/{model}",
                   hx_target="#downloaded_models",
                   hx_confirm=f"Are you sure you want to delete {model}?"),
                cls="flex items-center"
            ),
            cls="flex flex-col mb-4"
        ) for model, model_name in cached_models],
        id="downloaded_models"
    )

# Add a route to handle model deletion
@app.delete("/delete_model/{author}/{model_name}")
def delete_model(author: str, model_name: str):
    model_name = f"{author}/{model_name}"
    cache_info = scan_cache_dir()
    model_commit_hash = [revision.commit_hash
      for repo in cache_info.repos
      if repo.repo_id == model_name
      for revision in repo.revisions][0]
    try:
        print(model_commit_hash)
        delete_strategy = cache_info.delete_revisions(model_commit_hash)
        delete_strategy.execute()
        return list_downloaded_models()
    except Exception as e:
        return Div(f"Error deleting model: {str(e)}", id="downloaded_models")

# Add a route to handle model loading
@app.post("/load_model")
async def load_model_route(request: Request):
    global is_loading_model  # Access the loading flag
    is_loading_model = True  # Set the flag to true when loading starts
    try:
        form_data = await request.form()
        model_select = form_data.get("model_select")
        if not model_select:
            return Div("No model selected. Please select a model.", id="active_model_status")
        
        # Unload the current model if one is loaded
        global model, tokenizer
        if model is not None or tokenizer is not None:
            model = None
            tokenizer = None
        
        status = load_model(model_select)
        return Div(status, id="active_model_status")
    finally:
        is_loading_model = False  # Reset the flag when loading is done

# Add a route to handle model search
@app.post("/search_models")
def search_models(model_query: str):
    models = search_mlx_models(model_query)
    return Div(*[
        Div(
            Div(
                Span(model.split('/')[0], cls="block text-sm"),
                Span(model.split('/')[1], cls="block text-sm font-bold"),
                cls="flex-grow"
            ),
            Button("Download", 
                   hx_post=f"/download_model/{model}", 
                   hx_target="#download_status",
                   cls="btn btn-primary btn-xs ml-2"),
            cls="flex items-start justify-between mb-2 p-2 border-b border-gray-200"
        ) 
        for model in models
    ], id="model_list", cls="flex flex-col mb-4")

# Add a route to clear the search results
@app.post("/clear_search")
def clear_search():
    return Div(id="model_list")

# Add a route to clear the messages
@app.post("/clear_messages")
def clear_messages():
    global messages
    messages = [{"role": "system", "content": system_message}]
    return Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto")

# Modified stop_generation route
@app.post("/stop_generation")
async def stop_generation():
    global is_generating
    is_generating = False
    return ""  # Return an empty string instead of JSON

# Add a new route to handle settings updates
@app.post("/update_settings")
async def update_settings(request: Request):
    global temperature, max_tokens
    try:
        form_data = await request.form()
        temp = form_data.get('temp')
        tokens = form_data.get('tokens')
        
        if temp is not None:
            temperature = round(float(temp), 1)
        if tokens is not None:
            max_tokens = int(tokens)
        
        return Div(f"Settings updated: Temperature = {temperature}, Max Tokens = {max_tokens}", 
                   id="settings_status")
    except Exception as e:
        print(f"Error updating settings: {str(e)}")
        return Div(f"Error updating settings: {str(e)}", 
                   id="settings_status")

# Add a new route to handle system message updates
@app.post("/update_system_message")
async def update_system_message(request: Request):
    global system_message, messages
    try:
        form_data = await request.form()
        new_system_message = form_data.get('system_message')
        
        if new_system_message is not None:
            system_message = new_system_message
            # Update the system message in the messages list
            if messages and messages[0]['role'] == 'system':
                messages[0]['content'] = system_message
            else:
                messages.insert(0, {"role": "system", "content": system_message})
        
        return Div("System message updated successfully", 
                   id="system_message_status")
    except Exception as e:
        print(f"Error updating system message: {str(e)}")
        return Div(f"Error updating system message: {str(e)}", 
                   id="system_message_status")

# Run the chat model in a separate thread
@threaded
def get_response(r, idx):
    global model, tokenizer, messages, is_generating
    if model is None or tokenizer is None:
        messages[idx]["content"] = "Error: No model loaded. Please load a model first."
        messages[idx]["generating"] = False
        return
    
    is_generating = True
    for chunk in r:
        if not is_generating:
            break
        messages[idx]["content"] += chunk
    messages[idx]["generating"] = False
    is_generating = False

# Handle the form submission
@app.post("/")
def post(msg:str):
    global model, tokenizer, messages, is_generating, temperature, max_tokens, is_loading_model
    if model is None or tokenizer is None:
        return Div("Error: No model loaded. Please load a model first.", id="chatlist")
    
    if is_loading_model:  # Check if a model is currently loading
        return Div("Error: Model is currently loading. Please wait.", id="chatlist")
    
    idx = len(messages)
    messages.append({"role":"user", "content":msg.rstrip()})
    
    # Check if the system role is supported
    if "System role not supported" not in tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = tokenizer.apply_chat_template(messages[1:], tokenize=False, add_generation_prompt=True)  # Skip system message
    
    r = stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temperature)
    messages.append({"role":"assistant", "generating":True, "content":""})
    get_response(r, idx+1)
    
    # Create a list of all chat messages, skipping the system message
    chat_messages = [ChatMessage(i) for i in range(1, len(messages))]
    
    return (Div(*chat_messages,
                id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
            Script("""
                document.getElementById('msg-input').value = '';
                scrollToBottom();
                
                // Auto-resize textarea
                const textarea = document.getElementById('msg-input');
                textarea.style.height = 'auto';
                textarea.style.height = (textarea.scrollHeight) + 'px';
                
                // Add event listener for future inputs
                textarea.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';
                });
            """))

def run_app():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    run_app()