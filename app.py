from functools import lru_cache
import random
import os
from pathlib import Path
from io import BytesIO
import time
import logging
import json
import uuid
import websocket
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Basic configuration
COMFYUI_SERVER = os.getenv('COMFYUI_SERVER', 'http://127.0.0.1:8188')
WS_SERVER = os.getenv('WS_SERVER', 'ws://127.0.0.1:8188')
WORKFLOWS_DIR = "workflows"
DEFAULT_WORKFLOW = "dit_lora"

# Configure logging and HTTP session
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
http_session = requests.Session()
http_session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=100, max_retries=3))

# Pydantic models for request validation
class GenerateRequest(BaseModel):
    prompt: str
    workflow: str = DEFAULT_WORKFLOW
    width: int = Field(default=768, ge=64, le=2048)
    height: int = Field(default=768, ge=64, le=2048)

class HealthResponse(BaseModel):
    status: str
    comfyui_connected: bool

class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect_websocket(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(f"{WS_SERVER}/ws?clientId={self.client_id}")

    def disconnect_websocket(self):
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

    def queue_prompt(self, prompt):
        data = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode('utf-8')
        return http_session.post(f"{COMFYUI_SERVER}/prompt", data=data).json()

    def wait_for_completion(self, prompt_id):
        while True:
            message = json.loads(self.ws.recv())
            if message["type"] == "executing":
                if message["data"]["node"] is None and message["data"]["prompt_id"] == prompt_id:
                    break

    def get_image(self, prompt_id):
        history = http_session.get(f"{COMFYUI_SERVER}/history").json()
        if prompt_id not in history:
            raise ValueError("Prompt ID not found")
            
        for node_output in history[prompt_id]["outputs"].values():
            if "images" in node_output:
                image = node_output["images"][0]
                response = http_session.get(f"{COMFYUI_SERVER}/view?filename={image['filename']}")
                return {'content': response.content, 'filename': image['filename']}
        raise ValueError("No images found in output")

@lru_cache(maxsize=10)
def load_workflow_template(workflow_name: str):
    try:
        with open(os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json"), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading workflow '{workflow_name}': {str(e)}")

def customize_workflow(template: dict, prompt_text: str, width: int = 768, height: int = 768):
    workflow = template.copy()
    
    ksampler_node = next((node_id for node_id, node in workflow.items() 
                         if node.get("class_type") == "KSampler"), None)
    if not ksampler_node:
        raise ValueError("No KSampler node found")

    seed = random.randint(0, 0xffffffffffffffff)
    workflow[ksampler_node]["inputs"]["seed"] = seed
    workflow[ksampler_node]["inputs"]["steps"] = 20
    
    for node in workflow.values():
        if "width" in node.get("inputs", {}):
            node["inputs"]["width"] = width
        if "height" in node.get("inputs", {}):
            node["inputs"]["height"] = height
    
    for node_id, node in workflow.items():
        if (node.get("class_type") == "CLIPTextEncode" and 
            node.get("_meta", {}).get("title", "").lower().startswith("clip text encode (positive")):
            workflow[node_id]["inputs"]["text"] = prompt_text
            break
    
    return workflow, seed

app = FastAPI(title="Image Generation API", version="1.0.0")

@app.post("/generate")
async def generate(request: GenerateRequest):
    start_time = time.time()
    try:
        client = ComfyUIClient()
        client.connect_websocket()
        try:
            workflow, seed = customize_workflow(
                load_workflow_template(request.workflow), 
                request.prompt,
                request.width,
                request.height
            )
            
            result = client.queue_prompt(workflow)
            client.wait_for_completion(result['prompt_id'])
            image_data = client.get_image(result['prompt_id'])
            
            headers = {
                'X-Processing-Time': f"{time.time() - start_time:.2f}s",
                'X-Seed': str(seed),
                'X-Workflow': request.workflow,
                'X-Width': str(request.width),
                'X-Height': str(request.height),
                'Cache-Control': 'no-store'
            }
            
            return StreamingResponse(
                BytesIO(image_data['content']),
                media_type='image/png',
                headers=headers
            )

        finally:
            client.disconnect_websocket()

    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'processing_time': f"{processing_time:.2f}s"
            }
        )

@app.get("/workflows")
async def list_workflows():
    return {
        'workflows': [f.stem for f in Path(WORKFLOWS_DIR).glob("*.json")],
        'default': DEFAULT_WORKFLOW
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        status = http_session.get(f"{COMFYUI_SERVER}/history").status_code == 200
        return HealthResponse(
            status='healthy' if status else 'unhealthy',
            comfyui_connected=status
        )
    except:
        return HealthResponse(status='unhealthy', comfyui_connected=False)

if __name__ == '__main__':
    import uvicorn
    
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    logger.info("Starting FastAPI server on port 8787")
    uvicorn.run(app, host="0.0.0.0", port=8787)
