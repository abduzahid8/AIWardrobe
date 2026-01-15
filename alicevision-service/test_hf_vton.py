#!/usr/bin/env python3
"""Test Hugging Face Spaces IDM-VTON (FREE)"""
import os
os.chdir('/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service')

from gradio_client import Client, handle_file
import base64
import tempfile

# Create test images (small but valid)
print("Testing Hugging Face Spaces IDM-VTON...")

# Connect to a public IDM-VTON Space
print("Connecting to HF Space...")
try:
    # Try the official IDM-VTON space
    client = Client("yisol/IDM-VTON")
    print("Connected to yisol/IDM-VTON!")
    
    # Check available API
    print("\nAvailable API endpoints:")
    print(client.view_api())
    
except Exception as e:
    print(f"Error: {e}")
    
    # Try alternative spaces
    try:
        client = Client("levihsu/OOTDiffusion")
        print("Connected to levihsu/OOTDiffusion!")
        print(client.view_api())
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
