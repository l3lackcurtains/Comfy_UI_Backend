from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import websocket
import requests
import json
import uuid
import time
import os
import random
import logging
from io import BytesIO
from functools import lru_cache
from fastapi.responses import StreamingResponse

# Constants for service configuration
COMFYUI_SERVER = os.getenv('COMFYUI_SERVER', 'http://127.0.0.1:8188')
WS_SERVER = os.getenv('WS_SERVER', 'ws://127.0.0.1:8188')
WORKFLOWS_DIR = os.getenv('WORKFLOWS_DIR', 'workflows')
DEFAULT_WORKFLOW = os.getenv('DEFAULT_WORKFLOW', 'lora')

# Configure HTTP session for reuse
http_session = requests.Session()

class GenerateRequest(BaseModel):
    prompt: str
    workflow: str = DEFAULT_WORKFLOW
    width: int = Field(default=768, ge=64, le=2048)
    height: int = Field(default=768, ge=64, le=2048)

class HealthResponse(BaseModel):
    status: str
    comfyui_connected: bool

class ComfyUIClient:
    """Handles WebSocket communication with ComfyUI server"""
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
            if message["type"] == "executing" and message["data"]["node"] is None and message["data"]["prompt_id"] == prompt_id:
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
    """Loads and caches workflow templates from JSON files"""
    try:
        with open(os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json"), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading workflow '{workflow_name}': {str(e)}")

class ModelConfig:
    """Defines sampling parameters for different model types"""
    def __init__(self, steps, cfg, sampler, scheduler):
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler

# Predefined configurations for different model types
MODEL_CONFIGS = {
    "lora": ModelConfig(steps=60, cfg=7.0, sampler="dpmpp_2m", scheduler="karras"),
    "flux_dev": ModelConfig(steps=20, cfg=1, sampler="euler_ancestral", scheduler="simple"),
    "flux_schnell": ModelConfig(steps=8, cfg=1, sampler="euler", scheduler="simple")
}

def get_model_config(workflow_name: str) -> ModelConfig:
    """Returns appropriate model configuration based on workflow name"""
    base_name = workflow_name.split('.')[0].lower()
    return MODEL_CONFIGS.get(base_name, MODEL_CONFIGS["lora"])

def customize_workflow(template: dict, prompt_text: str, width: int = 768, height: int = 768, workflow_name: str = "lora"):
    """Customizes workflow template with user parameters and random seed"""
    workflow = template.copy()
    config = get_model_config(workflow_name)
    logging.info(f"Using configuration for model: {workflow_name}")

    ksampler_node_id = next((node_id for node_id, node in workflow.items()
                         if node.get("class_type") == "KSampler"), None)
    if not ksampler_node_id:
        raise ValueError("No KSampler node found")

    seed = random.randint(0, 0xffffffffffffffff)
    workflow[ksampler_node_id]["inputs"].update({
        "seed": seed,
        "steps": config.steps,
        "cfg": config.cfg,
        "sampler_name": config.sampler,
        "scheduler": config.scheduler,
    })

    # Update dimensions and prompt
    for node in workflow.values():
        if "width" in node.get("inputs", {}):
            node["inputs"]["width"] = width
        if "height" in node.get("inputs", {}):
            node["inputs"]["height"] = height
        if node.get("class_type") == "CLIPTextEncode":
            node["inputs"]["text"] = prompt_text
            break

    logging.info(f"Workflow configured: {workflow_name} ({width}x{height}, seed={seed})")
    return workflow, seed

def validate_workflow(workflow: dict, workflow_name: str) -> bool:
    """Ensures workflow contains all required nodes"""
    required_nodes = {"KSampler": False, "CLIPTextEncode": False, "VAEDecode": False, "CheckpointLoaderSimple": False}
    
    for node in workflow.values():
        node_type = node.get("class_type")
        if node_type in required_nodes:
            required_nodes[node_type] = True
    
    missing_nodes = [node for node, present in required_nodes.items() if not present]
    if missing_nodes:
        raise ValueError(f"Workflow {workflow_name} is missing required nodes: {', '.join(missing_nodes)}")
    return True

app = FastAPI(title="Image Generation API", version="1.0.0")

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Handles image generation requests"""
    start_time = time.time()
    try:
        logging.info(f"Processing generation request: workflow={request.workflow}, prompt={request.prompt}")
        client = ComfyUIClient()
        client.connect_websocket()
        try:
            workflow_template = load_workflow_template(request.workflow)
            validate_workflow(workflow_template, request.workflow)
            workflow, seed = customize_workflow(
                workflow_template,
                request.prompt,
                request.width,
                request.height,
                request.workflow
            )

            result = client.queue_prompt(workflow)
            if not result or 'prompt_id' not in result:
                raise ValueError("Failed to get valid prompt ID from server")

            prompt_id = result['prompt_id']
            client.wait_for_completion(prompt_id)
            image_data = client.get_image(prompt_id)

            return StreamingResponse(
                BytesIO(image_data['content']),
                media_type='image/png',
                headers={
                    'X-Processing-Time': f"{time.time() - start_time:.2f}s",
                    'X-Seed': str(seed),
                    'X-Workflow': request.workflow,
                    'X-Model-Config': request.workflow,
                    'X-Width': str(request.width),
                    'X-Height': str(request.height),
                    'Cache-Control': 'no-store'
                }
            )
        finally:
            client.disconnect_websocket()
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'processing_time': f"{time.time() - start_time:.2f}s"
            }
        )

@app.get("/workflows")
async def list_workflows():
    """Returns available workflows and default workflow"""
    return {
        'workflows': [f.stem for f in Path(WORKFLOWS_DIR).glob("*.json")],
        'default': DEFAULT_WORKFLOW
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Checks API and ComfyUI server health"""
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
    logging.info("Starting FastAPI server on port 8787")
    uvicorn.run(app, host="0.0.0.0", port=8787)
