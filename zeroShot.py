import cv2
import ollama
import base64
import io
from PIL import Image

def process_video_for_ollama(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Video could not be loaded or is empty.")
    
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    indices.sort()
    frames_base64 = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            with io.BytesIO() as output:
                pil_img.save(output, format="JPEG", quality=85)
                frames_base64.append(base64.b64encode(output.getvalue()).decode('utf-8'))
    
    cap.release()
    print(f"Extracted {len(frames_base64)} frames")
    return frames_base64

MODEL_NAME = "qwen3-vl:4b" 
VIDEO_PATH = "Timeline 1.mp4"

print(f"Extracting frames from {VIDEO_PATH}...")
try:
    video_frames = process_video_for_ollama(VIDEO_PATH, num_frames=3)
    
    # TEST 1: Einfachster Prompt
    print("\n=== TEST 1: Simple Description ===")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'user',
                'content': 'Describe what you see in one sentence.',
                'images': video_frames
            }
        ],
        stream=False,
        options={
            'temperature': 0.3,  
            'num_predict': 50,  # Sehr kurz halten
        }
    )
    
    print("Response:", response.get('message', {}).get('content', 'NO CONTENT'))
    print("Thinking:", response.get('message', {}).get('thinking', 'NO THINKING')[:200])
    
    # TEST 2: Mit System Prompt
    print("\n=== TEST 2: With System Prompt ===")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'system',
                'content': 'You are a video analysis expert. Provide concise, direct answers.'
            },
            {
                'role': 'user',
                'content': 'Look at these ego cooking video frames. What is the person doing?',
                'images': video_frames
            }
        ],
        stream=False,
        options={
            'temperature': 0.3,  
            'num_predict': 50,
        }
    )
    
    print("Response:", response.get('message', {}).get('content', 'NO CONTENT'))
    print("Thinking:", response.get('message', {}).get('thinking', 'NO THINKING')[:200])
    
    # TEST 3: Nur 1 Frame
    print("\n=== TEST 3: Single Frame ===")
    single_frame = process_video_for_ollama(VIDEO_PATH, num_frames=1)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'user',
                'content': 'What action is shown? Answer in 2-3 words only.',
                'images': single_frame
            }
        ],
        stream=False,
        options={
            'temperature': 0.1,  
            'num_predict': 20,
        }
    )
    
    print("Response:", response.get('message', {}).get('content', 'NO CONTENT'))
    print("Thinking:", response.get('message', {}).get('thinking', 'NO THINKING')[:200])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()