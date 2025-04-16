from flask import Flask, jsonify, request, send_file
import json
import requests
import websocket
import uuid
import logging
import time
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

COMFYUI_SERVER = "http://127.0.0.1:8188"
WS_SERVER = "ws://127.0.0.1:8188"

class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.websocket = None

    def connect_websocket(self):
        """Establish WebSocket connection"""
        if self.websocket is None:
            logger.info("Establishing WebSocket connection")
            self.websocket = websocket.WebSocket()
            self.websocket.connect(f"{WS_SERVER}/ws?clientId={self.client_id}")
            logger.info("WebSocket connection established")

    def disconnect_websocket(self):
        """Close WebSocket connection"""
        if self.websocket:
            self.websocket.close()
            self.websocket = None
            logger.info("WebSocket connection closed")

    def queue_prompt(self, prompt):
        """Queue a prompt to ComfyUI"""
        logger.info("Queueing prompt")
        data = json.dumps({
            "prompt": prompt,
            "client_id": self.client_id
        }).encode('utf-8')
        
        response = requests.post(f"{COMFYUI_SERVER}/prompt", data=data)
        if response.status_code != 200:
            raise Exception(f"Failed to queue prompt: {response.text}")
        
        result = response.json()
        logger.info(f"Prompt queued with ID: {result.get('prompt_id')}")
        return result

    def get_image(self, prompt_id):
        """Get image from history"""
        logger.info(f"Fetching image for prompt {prompt_id}")
        response = requests.get(f"{COMFYUI_SERVER}/history/{prompt_id}")
        if response.status_code != 200:
            raise Exception("Failed to get history")

        history = response.json()
        
        # Find the SaveImage node output
        for node_id, node_output in history[prompt_id]['outputs'].items():
            if 'images' in node_output:
                image_data = node_output['images'][0]
                image_url = f"{COMFYUI_SERVER}/view?filename={image_data['filename']}&type=output"
                logger.info(f"Found image: {image_data['filename']}")
                
                # Get the image content
                image_response = requests.get(image_url)
                if image_response.status_code != 200:
                    raise Exception("Failed to download image")
                    
                return {
                    'content': image_response.content,
                    'filename': image_data['filename']
                }
        
        raise Exception("No image found in output")

    def wait_for_completion(self, prompt_id):
        """Wait for prompt execution to complete"""
        while True:
            try:
                out = self.websocket.recv()
                if not isinstance(out, str):
                    continue
                
                message = json.loads(out)
                if not isinstance(message, dict):
                    continue

                msg_type = message.get('type')
                if msg_type == 'executing':
                    data = message.get('data', {})
                    if not data.get('node') and data.get('prompt_id') == prompt_id:
                        return True
                    
            except websocket.WebSocketConnectionClosedException:
                raise
            except Exception:
                continue

def load_workflow(workflow_path, prompt_text=None):
    """Load and customize workflow"""
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
            logger.info(f"Loaded workflow from {workflow_path}")

        if prompt_text:
            # Assuming node 6 is CLIPTextEncode for positive prompt
            workflow["6"]["inputs"]["text"] = prompt_text
            logger.info(f"Updated positive prompt: {prompt_text}")

        # Always generate random seed
        import random
        seed = random.randint(0, 0xffffffffffffffff)
        workflow["3"]["inputs"]["seed"] = seed
        logger.info(f"Generated random seed: {seed}")

        return workflow, seed
    except Exception as e:
        logger.error(f"Failed to load workflow: {str(e)}")
        raise

@app.route('/generate', methods=['POST'])
def generate():
    """Generate image endpoint"""
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        prompt_text = data['prompt']
        logger.info(f"Received generation request with prompt: {prompt_text}")

        # Initialize ComfyUI client
        client = ComfyUIClient()
        client.connect_websocket()

        try:
            # Load and customize workflow
            workflow, used_seed = load_workflow('dit_loraapi.json', prompt_text)

            # Queue the prompt
            result = client.queue_prompt(workflow)
            prompt_id = result['prompt_id']

            # Wait for completion
            client.wait_for_completion(prompt_id)

            # Get the generated image
            image_data = client.get_image(prompt_id)
            
            # Create BytesIO object from image content
            image_bytes = BytesIO(image_data['content'])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response with custom headers
            response = send_file(
                image_bytes,
                mimetype='image/png',
                as_attachment=True,
                download_name=image_data['filename']
            )
            response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
            response.headers['X-Seed'] = str(used_seed)
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
    """Health check endpoint"""
    try:
        response = requests.get(f"{COMFYUI_SERVER}/history")
        if response.status_code == 200:
            return jsonify({'status': 'healthy', 'comfyui_connected': True})
        else:
            return jsonify({'status': 'unhealthy', 'comfyui_connected': False}), 503
    except:
        return jsonify({'status': 'unhealthy', 'comfyui_connected': False}), 503

if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
