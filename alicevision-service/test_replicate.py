#!/usr/bin/env python3
import os
os.chdir('/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service')

from dotenv import load_dotenv
load_dotenv('.env')

import replicate

# Using a more realistic test image
person_uri = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
garment_uri = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYGD4HwABBAEAr4J5egAAAABJRU5ErkJggg=='

# Test WITHOUT version (use latest)
print('Test 1: cuuupid/idm-vton (no version)...')
try:
    output = replicate.run(
        'cuuupid/idm-vton',
        input={
            'human_img': person_uri,
            'garm_img': garment_uri,
            'garment_des': 'A shirt',
            'category': 'upper_body'
        }
    )
    print(f'SUCCESS! Output: {output}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')

print('\nTest 2: viktorfa/oot_diffusion (no version)...')
try:
    output = replicate.run(
        'viktorfa/oot_diffusion',
        input={
            'model_image': person_uri,
            'garment_image': garment_uri,
            'steps': 10,
            'guidance_scale': 2.5,
            'garment_type': 'upperbody'
        }
    )
    print(f'SUCCESS! Output: {output}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
