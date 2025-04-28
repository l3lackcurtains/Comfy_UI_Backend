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

COMFYUI_SERVER = os.getenv('COMFYUI_SERVER', 'http://127.0.0.1:8188')
WS_SERVER = os.getenv('WS_SERVER', 'ws://127.0.0.1:8188')
WORKFLOWS_DIR = "workflows"
DEFAULT_WORKFLOW = "lora"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
http_session = requests.Session()
http_session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=100, max_retries=3))

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
    def __init__(self, steps, cfg, sampler, scheduler, denoise=1.0):
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.denoise = denoise

MODEL_CONFIGS = {
    "lora": ModelConfig(
        steps=80,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras"
    ),
    "flux_dev": ModelConfig(
        steps=20,
        cfg=1,
        sampler="euler_ancestral",
        scheduler="simple"
    ),
    "flux_schnell": ModelConfig(
        steps=12,
        cfg=1,
        sampler="euler_ancestral",
        scheduler="simple"
    )
}

def get_model_config(workflow_name: str) -> ModelConfig:
    base_name = workflow_name.split('.')[0].lower()
    return MODEL_CONFIGS.get(base_name, MODEL_CONFIGS["lora"])

def customize_workflow(template: dict, prompt_text: str, width: int = 768, height: int = 768, workflow_name: str = "lora"):
    workflow = template.copy()
    config = get_model_config(workflow_name)
    logging.info(f"Using configuration for model: {workflow_name}")

    # Find KSampler node
    ksampler_node_id = next((node_id for node_id, node in workflow.items()
                             if node.get("class_type") == "KSampler"), None)

    if not ksampler_node_id:
        raise ValueError("No KSampler node found")

    ksampler_node = workflow[ksampler_node_id]

    # Generate random seed
    seed = random.randint(0, 0xffffffffffffffff)
    
    # Update KSampler parameters
    ksampler_node["inputs"].update({
        "seed": seed,
        "steps": config.steps,
        "cfg": config.cfg,
        "sampler_name": config.sampler,
        "scheduler": config.scheduler,
        "denoise": config.denoise,
    })

    # Update dimensions for all nodes that have width/height inputs
    for node in workflow.values():
        if "width" in node.get("inputs", {}):
            node["inputs"]["width"] = width
        if "height" in node.get("inputs", {}):
            node["inputs"]["height"] = height

    # Find and update the first CLIPTextEncode node with the prompt
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode":
            node["inputs"]["text"] = prompt_text
            break

    logging.info(f"Workflow configuration summary for {workflow_name}:")
    logging.info(f"- Seed: {seed}")
    logging.info(f"- Dimensions: {width}x{height}")
    logging.info(f"- Sampling settings: steps={config.steps}, cfg={config.cfg}, "
                f"sampler={config.sampler}, scheduler={config.scheduler}")

    return workflow, seed

def validate_workflow(workflow: dict, workflow_name: str) -> bool:
    required_nodes = {
        "KSampler": False,
        "CLIPTextEncode": False,
        "VAEDecode": False,
        "CheckpointLoaderSimple": False
    }
    
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
    start_time = time.time()
    try:
        logging.info(f"Received generation request for workflow {request.workflow} with prompt: {request.prompt}")
        client = ComfyUIClient()
        client.connect_websocket()
        try:
            workflow_template = load_workflow_template(request.workflow)
            logging.info(f"Loaded workflow template: {request.workflow}")

            validate_workflow(workflow_template, request.workflow)

            # Decide if you want to use a fixed seed for debugging
            use_fixed = False # Set to True to use the fixed seed below for testing
            fixed_seed = 806955505434698 # The seed from lora.json

            workflow, seed = customize_workflow(
                workflow_template,
                request.prompt,
                request.width,
                request.height,
                request.workflow
            )

            logging.info(f"Customized workflow with prompt. Sending to ComfyUI...")
            # Optional: Log the final workflow being sent
            # logging.debug(f"Workflow being sent: {json.dumps(workflow, indent=2)}")
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
                    'X-Model-Config': request.workflow, # Consider reflecting actual config used
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
