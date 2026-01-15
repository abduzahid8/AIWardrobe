#!/usr/bin/env python3
"""Full integration test for the updated VTON module"""
import os
import sys
os.chdir('/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service')
sys.path.insert(0, '/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service')

from dotenv import load_dotenv
load_dotenv('.env')

# Now test the module
from modules.catvton_tryon import get_vton_engine

# Minimal test images
person_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
garment_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYGD4HwABBAEAr4J5egAAAABJRU5ErkJggg=='

print("Testing updated VTON with FREE HF Spaces...")
engine = get_vton_engine()
result = engine.try_on(person_b64, garment_b64, "upper_body")

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Method: {result.method_used}")
print(f"  Time: {result.processing_time_ms:.0f}ms")
print(f"  Has Image: {len(result.result_image_b64) > 0}")
