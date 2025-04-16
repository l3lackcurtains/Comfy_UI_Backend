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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores

COMFYUI_SERVER = "http://127.0.0.1:8188"
WS_SERVER = "ws://127.0.0.1:8188"

# Configure requests session for connection pooling
http_session = requests.Session()
http_session.mount('http://', requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=100,
    max_retries=3,
    pool_block=False
))

class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.websocket = None
        self._workflow_cache = {}

    def connect_websocket(self):
        if self.websocket is None:
            self.websocket = websocket.WebSocket()
            self.websocket.connect(f"{WS_SERVER}/ws?clientId={self.client_id}")

    def disconnect_websocket(self):
        if self.websocket:
            self.websocket.close()
            self.websocket = None

    def queue_prompt(self, prompt):
        data = json.dumps({
            "prompt": prompt,
            "client_id": self.client_id
        }).encode('utf-8')
        
        response = http_session.post(f"{COMFYUI_SERVER}/prompt", data=data)
        response.raise_for_status()
        return response.json()

    def get_image(self, prompt_id):
        response = http_session.get(f"{COMFYUI_SERVER}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()
        
        for node_output in history[prompt_id]['outputs'].values():
            if 'images' in node_output:
                image_data = node_output['images'][0]
                image_url = f"{COMFYUI_SERVER}/view?filename={image_data['filename']}&type=output"
                
                image_response = http_session.get(image_url)
                image_response.raise_for_status()
                
                return {
                    'content': image_response.content,
                    'filename': image_data['filename']
                }
        
        raise Exception("No image found in output")

    def wait_for_completion(self, prompt_id):
        while True:
            try:
                out = self.websocket.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if isinstance(message, dict):
                        if (message.get('type') == 'executing' and 
                            not message.get('data', {}).get('node') and 
                            message.get('data', {}).get('prompt_id') == prompt_id):
                            return True
            except websocket.WebSocketConnectionClosedException:
                raise
            except Exception:
                continue

@lru_cache(maxsize=10)
def load_workflow_template(workflow_path):
    """Cache the base workflow template"""
    with open(workflow_path, 'r') as f:
        return json.load(f)

def customize_workflow(template, prompt_text=None):
    """Customize workflow with minimal steps for fastest generation"""
    workflow = template.copy()
    
    if prompt_text:
        workflow["6"]["inputs"]["text"] = prompt_text
    
    # Set ultra-fast sampling parameters
    workflow["3"]["inputs"]["seed"] = random.randint(0, 0xffffffffffffffff)
    workflow["3"]["inputs"]["steps"] = 10  # Minimum recommended steps
    workflow["3"]["inputs"]["sampler_name"] = "euler_ancestral"
    workflow["3"]["inputs"]["scheduler"] = "karras"
    workflow["3"]["inputs"]["cfg"] = 5.0  # Minimum recommended CFG
    
    return workflow

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        prompt_text = data['prompt']
        client = ComfyUIClient()
        client.connect_websocket()

        try:
            # Load cached workflow template and customize
            template = load_workflow_template('dit_loraapi.json')
            workflow, used_seed = customize_workflow(template, prompt_text), template["3"]["inputs"]["seed"]

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
                'Cache-Control': 'no-store'
            })
            return response

        finally:
            client.disconnect_websocket()

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
    logger.info("Starting Flask server on port 5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
