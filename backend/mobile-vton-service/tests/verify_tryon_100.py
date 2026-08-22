import time
import requests
import json
import base64

# Small 1x1 base64 transparent PNG
DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

MODAL_URL = "https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run"

def test_tryon():
    payload = {
        "person_image": DUMMY_IMAGE,
        "garment_image": DUMMY_IMAGE,
        "garment_description": "A white t-shirt",
        "num_inference_steps": 1,
        "seed": 42
    }

    print("Sending request to Modal GPU endpoint...")
    start_time = time.time()
    
    try:
        response = requests.post(MODAL_URL, json=payload, timeout=240)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            success = data.get("success", False)
            print(f"✅ Success! Elapsed: {elapsed:.2f}s | HTTP 200")
            print(f"Response: {json.dumps(data)[:150]}...")
            return True
        else:
            print(f"❌ Failed! Elapsed: {elapsed:.2f}s | HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Network/Timeout Error: {e}")
        return False

if __name__ == "__main__":
    print("--- Running VTON Stress Test ---")
    for i in range(3):
        print(f"\nTest {i+1}/3:")
        test_tryon()
        time.sleep(2)
