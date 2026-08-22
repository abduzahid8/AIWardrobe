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
garment_image = to_base64('/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/basic_white_tshirt.png')

payload = {
    'person_image': person_image,
    'garment_image': garment_image,
    'garment_description': 'white tshirt',
    'guidance_scale': 7.5,
    'num_inference_steps': 25,
    'seed': 42
}

print('Testing Mobile VTON single try-on...')
print('URL: https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run

req = urllib.request.Request(
    'https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read().decode())

    print('\n=== RESULT ===')
    print(f'Success: {data.get("success")}')
    print(f'Method: {data.get("method_used")}')
    print(f'Elapsed: {data.get("elapsed_ms")} ms')
    print(f'Result image length: {len(data.get("result_image", ""))} chars')

    if data.get('success'):
        print('\n✓ Mobile VTON is working correctly')
        # Save result image to file
        result_image = data.get('result_image', '')
        if result_image and result_image.startswith('data:'):
            header, b64data = result_image.split(',', 1)
            image_data = base64.b64decode(b64data)
            with open('/Users/zohidvohidjonov/Desktop/AIWardrobe/mobile-vton-service/test_result.png', 'wb') as f:
                f.write(image_data)
            print('✓ Result image saved to test_result.png')
    else:
        print('\n✗ Mobile VTON failed')
except urllib.error.URLError as e:
    print(f'\n✗ Error: {e.reason}')
