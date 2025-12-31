#!/bin/bash
# Setup script for Grounded SAM2 + FashionCLIP models
# Run this after installing requirements.txt

set -e

echo "🚀 Setting up Grounded SAM2 + FashionCLIP models..."
echo ""

# Create weights directory
mkdir -p weights
cd weights

# Download Grounding DINO weights
echo "📥 Downloading Grounding DINO model..."
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    curl -L --progress-bar https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o groundingdino_swint_ogc.pth
    echo "✅ Grounding DINO downloaded"
else
    echo "⏭️  Grounding DINO already exists"
fi

# Download SAM2 weights
echo "📥 Downloading SAM2 base model..."
if [ ! -f "sam2_hiera_base.pt" ]; then
    curl -L --progress-bar https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -o sam2_hiera_base.pt
    echo "✅ SAM2 base downloaded"
else
    echo "⏭️  SAM2 base already exists"
fi

# Optional: Download SAM2 large model for better accuracy
echo ""
echo "📥 Downloading SAM2 large model (optional, better accuracy)..."
if [ ! -f "sam2_hiera_large.pt" ]; then
    curl -L --progress-bar https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -o sam2_hiera_large.pt
    echo "✅ SAM2 large downloaded"
else
    echo "⏭️  SAM2 large already exists"
fi

cd ..

# Clone config files for Grounding DINO if needed
if [ ! -d "groundingdino" ]; then
    echo ""
    echo "📥 Cloning Grounding DINO configs..."
    git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git groundingdino_repo
    cp -r groundingdino_repo/groundingdino ./
    rm -rf groundingdino_repo
    echo "✅ Grounding DINO configs ready"
fi

# Clone SAM2 configs if needed
if [ ! -d "sam2" ]; then
    echo ""
    echo "📥 Cloning SAM2 configs..."
    git clone --depth 1 https://github.com/facebookresearch/segment-anything-2.git sam2_repo
    cp -r sam2_repo/sam2 ./
    rm -rf sam2_repo
    echo "✅ SAM2 configs ready"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "📊 Model sizes:"
ls -lh weights/
echo ""
echo "🎯 You can now use:"
echo "   - Grounded SAM2 for text-prompted segmentation"
echo "   - FashionCLIP for attribute extraction"
echo ""
echo "⚡ Start the service with: python main.py"
