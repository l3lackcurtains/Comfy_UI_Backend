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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Basic configuration
COMFYUI_SERVER = os.getenv('COMFYUI_SERVER', 'http://127.0.0.1:8188')
WS_SERVER = os.getenv('WS_SERVER', 'ws://127.0.0.1:8188')
WORKFLOWS_DIR = "workflows"
DEFAULT_WORKFLOW = "lora"

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

class ModelConfig:
    def __init__(self, steps=20, cfg=7.0, sampler="euler_a", scheduler="normal", denoise=1.0,
                 lora_strength_model=1.0, lora_strength_clip=1.0, custom_params=None):
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.denoise = denoise
        self.lora_strength_model = lora_strength_model
        self.lora_strength_clip = lora_strength_clip
        self.custom_params = custom_params or {}

MODEL_CONFIGS = {
    "lora": ModelConfig(
        steps=30,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras",
        lora_strength_model=0.75,
        lora_strength_clip=1.0
    ),
    
    "lora_1": ModelConfig(
        steps=30,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras",
        lora_strength_model=0.75,
        lora_strength_clip=1.0
    ),
    
    "flux_dev": ModelConfig(
        steps=20,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        denoise=1.0,
        custom_params={
            "width": 768,
            "height": 768
        }
    ),
    
    "flux_schnell": ModelConfig(
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        denoise=1.0,
        custom_params={
            "width": 768,
            "height": 768
        }
    )
}

def get_model_config(workflow_name: str) -> ModelConfig:
    """Get the configuration for a specific model/workflow."""
    base_name = workflow_name.split('.')[0].lower()
    return MODEL_CONFIGS.get(base_name, MODEL_CONFIGS["lora"])  # Default to lora config

def customize_workflow(template: dict, prompt_text: str, width: int = 768, height: int = 768, workflow_name: str = "lora"):
    workflow = template.copy()
    
    # Get model-specific configuration
    config = get_model_config(workflow_name)
    logging.info(f"Using configuration for model: {workflow_name}")
    
    # Find key nodes
    ksampler_node = next((node_id for node_id, node in workflow.items() 
                         if node.get("class_type") == "KSampler"), None)
    lora_loader_node = next((node_id for node_id, node in workflow.items() 
                           if node.get("class_type") in ["LoraLoader", "DiffControlNetLoader"]), None)
    
    if not ksampler_node:
        raise ValueError("No KSampler node found")

    # Set sampling parameters based on model config
    seed = random.randint(0, 0xffffffffffffffff)
    workflow[ksampler_node]["inputs"].update({
        "seed": seed,
        "steps": config.steps,
        "cfg": config.cfg,
        "sampler_name": config.sampler,
        "scheduler": config.scheduler,
        "denoise": config.denoise,
    })
    
    # Configure LoRA strength if present
    if lora_loader_node and workflow_name.startswith("lora"):
        workflow[lora_loader_node]["inputs"].update({
            "strength_model": config.lora_strength_model,
            "strength_clip": config.lora_strength_clip,
        })
        logging.info(f"Configured LoRA node {lora_loader_node} with strengths: "
                    f"model={config.lora_strength_model}, clip={config.lora_strength_clip}")
    
    # Apply model-specific custom parameters
    if config.custom_params:
        for node in workflow.values():
            if "inputs" in node:
                for param_name, param_value in config.custom_params.items():
                    if param_name in node["inputs"]:
                        node["inputs"][param_name] = param_value
                        logging.info(f"Set custom parameter {param_name}={param_value} for node {node.get('class_type')}")
    
    # Update dimensions
    for node in workflow.values():
        if "width" in node.get("inputs", {}):
            node["inputs"]["width"] = width
        if "height" in node.get("inputs", {}):
            node["inputs"]["height"] = height
    
    # Handle prompts
    prompt_set = False
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            if ("positive" in str(node.get("_meta", {}).get("title", "")).lower() or 
                not "negative" in str(node.get("_meta", {}).get("title", "")).lower()):
                workflow[node_id]["inputs"]["text"] = prompt_text
                prompt_set = True
                logging.info(f"Set positive prompt in node {node_id}")
                
                # Handle negative prompt node if present
                negative_node = next((n_id for n_id, n in workflow.items() 
                                   if n.get("class_type") == "CLIPTextEncode" and 
                                   "negative" in str(n.get("_meta", {}).get("title", "")).lower()), None)
                if negative_node:
                    workflow[negative_node]["inputs"]["text"] = ""
                    logging.info(f"Set empty negative prompt in node {negative_node}")
    
    if not prompt_set:
        raise ValueError("No suitable CLIP text encode node found in workflow")
    
    # Log configuration summary
    logging.info(f"Workflow configuration summary for {workflow_name}:")
    logging.info(f"- Seed: {seed}")
    logging.info(f"- Dimensions: {width}x{height}")
    logging.info(f"- Sampling settings: steps={config.steps}, cfg={config.cfg}, "
                f"sampler={config.sampler}, scheduler={config.scheduler}")
    if config.custom_params:
        logging.info(f"- Custom parameters: {config.custom_params}")
    
    return workflow, seed

def validate_workflow(workflow: dict, workflow_name: str) -> bool:
    """Validates that a workflow has all necessary components for the specified model."""
    required_nodes = {
        "lora": {
            "LoraLoader": False,
            "KSampler": False,
            "CLIPTextEncode": False,
            "VAEDecode": False,
            "CheckpointLoaderSimple": False,
        },
        "lora_1": {
            "LoraLoader": False,
            "KSampler": False,
            "CLIPTextEncode": False,
            "VAEDecode": False,
            "CheckpointLoaderSimple": False,
        },
        "flux_dev": {
            "KSampler": False,
            "CLIPTextEncode": False,
            "VAEDecode": False,
            "CheckpointLoaderSimple": False,
        },
        "flux_schnell": {
            "KSampler": False,
            "CLIPTextEncode": False,
            "VAEDecode": False,
            "CheckpointLoaderSimple": False,
        }
    }
    
    # Use lora requirements as default
    model_type = next((k for k in required_nodes.keys() if workflow_name.startswith(k)), "lora")
    required = required_nodes[model_type]
    
    for node in workflow.values():
        node_type = node.get("class_type")
        if node_type in required:
            required[node_type] = True
    
    missing_nodes = [node for node, present in required.items() if not present]
    if missing_nodes:
        raise ValueError(f"Workflow {workflow_name} is missing required nodes: {', '.join(missing_nodes)}")
    
    return True

app = FastAPI(title="Image Generation API", version="1.0.0")

@app.post("/generate")
async def generate(request: GenerateRequest):
    start_time = time.time()
    try:
        logging.info(f"Received generation request for workflow {request.workflow} with prompt: {request.prompt}")
        client = ComfyUIClient()
        client.connect_websocket()
        try:
            workflow_template = load_workflow_template(request.workflow)
            logging.info(f"Loaded workflow template: {request.workflow}")
            
            # Validate workflow before proceeding
            validate_workflow(workflow_template, request.workflow)
            
            workflow, seed = customize_workflow(
                workflow_template, 
                request.prompt,
                request.width,
                request.height,
                request.workflow
            )
            
            logging.info(f"Customized workflow with prompt. Sending to ComfyUI...")
            result = client.queue_prompt(workflow)
            
            if not result or 'prompt_id' not in result:
                raise ValueError("Failed to get valid prompt ID from server")
                
            prompt_id = result['prompt_id']
            logging.info(f"Prompt queued with ID: {prompt_id}")
            
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
