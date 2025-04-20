from flask import Flask, jsonify, request, send_file
import json
import requests
import websocket
import uuid
import logging
import time
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import random
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect_websocket(self):
        """Connect to ComfyUI websocket"""
        self.ws = websocket.WebSocket()
        self.ws.connect(f"{WS_SERVER}/ws?clientId={self.client_id}")

    def disconnect_websocket(self):
        """Safely disconnect websocket"""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None

    def queue_prompt(self, prompt):
        """Queue a prompt for processing"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        response = http_session.post(f"{COMFYUI_SERVER}/prompt", data=data)
        return response.json()

    def wait_for_completion(self, prompt_id):
        """Wait for prompt processing to complete"""
        while True:
            out = self.ws.recv()
            if out is None:
                continue

            try:
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # Execution complete
            except Exception as e:
                logger.error(f"Error parsing websocket message: {e}")
                continue

    def get_image(self, prompt_id):
        """Get the generated image data"""
        response = http_session.get(f"{COMFYUI_SERVER}/history")
        history = response.json()
        
        if prompt_id not in history:
            raise ValueError("Prompt ID not found in history")
            
        outputs = history[prompt_id]["outputs"]
        if not outputs:
            raise ValueError("No outputs found for prompt")
            
        # Find the first node with images
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                image_data = node_output["images"][0]
                
                # Get the actual image data
                image_response = http_session.get(f"{COMFYUI_SERVER}/view?filename={image_data['filename']}")
                return {
                    'content': image_response.content,
                    'filename': image_data['filename']
                }
                
        raise ValueError("No images found in output")

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores

COMFYUI_SERVER = os.getenv('COMFYUI_SERVER', 'http://127.0.0.1:8188')
WS_SERVER = os.getenv('WS_SERVER', 'ws://127.0.0.1:8188')
WORKFLOWS_DIR = "workflows"  # Directory containing workflow templates
DEFAULT_WORKFLOW = "dit_lora"

# Ensure workflows directory exists
os.makedirs(WORKFLOWS_DIR, exist_ok=True)

# Configure requests session for connection pooling
http_session = requests.Session()
http_session.mount('http://', requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=100,
    max_retries=3,
    pool_block=False
))

def get_available_workflows():
    """Get list of available workflow templates"""
    workflows = []
    for file in Path(WORKFLOWS_DIR).glob("*.json"):
        workflows.append(file.stem)
    return workflows

@lru_cache(maxsize=10)
def load_workflow_template(workflow_name):
    """Cache the workflow template"""
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json")
    try:
        with open(workflow_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Workflow template '{workflow_name}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in workflow template '{workflow_name}'")

def customize_workflow(template, prompt_text=None):
    """Customize workflow with minimal steps for fastest generation"""
    workflow = template.copy()
    
    # Find KSampler node ID
    ksampler_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            ksampler_node = node_id
            break
    
    if not ksampler_node:
        raise ValueError("No KSampler node found in workflow")

    # Set minimal sampling parameters that work for both Flux and LoRA
    workflow[ksampler_node]["inputs"]["seed"] = random.randint(0, 0xffffffffffffffff)
    workflow[ksampler_node]["inputs"]["steps"] = 20
    
    # Update prompt if provided
    if prompt_text:
        for node_id, node in workflow.items():
            if (node.get("class_type") == "CLIPTextEncode" and 
                node.get("_meta", {}).get("title", "").lower().startswith("clip text encode (positive")):
                workflow[node_id]["inputs"]["text"] = prompt_text
                break
    
    return workflow, workflow[ksampler_node]["inputs"]["seed"]

@app.route('/workflows', methods=['GET'])
def list_workflows():
    """Endpoint to list available workflows"""
    return jsonify({
        'workflows': get_available_workflows(),
        'default': DEFAULT_WORKFLOW
    })

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        prompt_text = data['prompt']
        workflow_name = data.get('workflow', DEFAULT_WORKFLOW)

        client = ComfyUIClient()
        client.connect_websocket()

        try:
            # Load cached workflow template and customize
            template = load_workflow_template(workflow_name)
            workflow, used_seed = customize_workflow(template, prompt_text)  # Fixed tuple unpacking

            # Queue and process
            result = client.queue_prompt(workflow)
            client.wait_for_completion(result['prompt_id'])
            image_data = client.get_image(result['prompt_id'])
            
            # Prepare response
            response = send_file(
                BytesIO(image_data['content']),
                mimetype='image/png',
                as_attachment=True,
                download_name=image_data['filename']
            )
            
            processing_time = time.time() - start_time
            response.headers.update({
                'X-Processing-Time': f"{processing_time:.2f}s",
                'X-Seed': str(used_seed),
                'X-Workflow': workflow_name,
                'Cache-Control': 'no-store'
            })
            return response

        finally:
            client.disconnect_websocket()

    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = jsonify({
            'success': False,
            'error': str(e),
            'processing_time': f"{processing_time:.2f}s"
        })
        error_response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
        return error_response, 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        response = http_session.get(f"{COMFYUI_SERVER}/history")
        return jsonify({
            'status': 'healthy', 
            'comfyui_connected': response.status_code == 200
        }), 200 if response.status_code == 200 else 503
    except:
        return jsonify({
            'status': 'unhealthy', 
            'comfyui_connected': False
        }), 503

if __name__ == '__main__':
    logger.info("Starting Flask server on port 8787")
    app.run(host='0.0.0.0', port=8787, threaded=True)
