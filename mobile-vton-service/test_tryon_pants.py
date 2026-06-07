import base64
import json
import urllib.request
import urllib.error

def to_base64(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    ext = file_path.split('.')[-1]
    mime = 'image/jpeg' if ext in ['jpg', 'jpeg'] else 'image/png'
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

person_image = to_base64('/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/mannequin_front.png')
garment_image = to_base64('/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/basic_brown_pants.png')

garments = [{'garment_image': garment_image, 'description': 'brown pants', 'label': 'pants'}]

payload = {
    'person_image': person_image,
    'garments': garments,
    'guidance_scale': 7.5,
    'num_inference_steps': 25,
    'seed': 42,
    'pipeline_version': 'fused_v2'
}

print('Testing brown pants via /tryon/multi-fused...')
req = urllib.request.Request(
    'https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read().decode())
    print(f"Success: {data.get('success')}  Elapsed: {data.get('elapsed_ms')} ms")
    print(f"Rendered: {data.get('rendered_garments')}")
    if data.get('success'):
        result_image = data.get('result_image', '')
        if result_image.startswith('data:'):
            header, b64data = result_image.split(',', 1)
            image_data = base64.b64decode(b64data)
            with open('/Users/zohidvohidjonov/Desktop/AIWardrobe/mobile-vton-service/test_pants_result.png', 'wb') as f:
                f.write(image_data)
            print('Saved to test_pants_result.png')
except urllib.error.URLError as e:
    print(f'Error: {e.reason}')
