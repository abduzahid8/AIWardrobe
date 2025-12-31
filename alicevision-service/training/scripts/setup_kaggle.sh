#!/bin/bash
# =============================================================================
# AIWardrobe - Kaggle Setup Script
# Download iMaterialist Fashion 2019 (1M+ clothing images)
# =============================================================================

echo "🎨 AIWardrobe - Kaggle Dataset Setup"
echo "======================================"

# Check if kaggle.json exists
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle API key found!"
else
    echo ""
    echo "❌ Kaggle API key not found!"
    echo ""
    echo "📝 Follow these steps:"
    echo ""
    echo "1. Go to: https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New Token'"
    echo "4. Download kaggle.json"
    echo "5. Run these commands:"
    echo ""
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "6. Then run this script again!"
    exit 1
fi

# Install kaggle if needed
pip3 install kaggle -q

# Create dataset directory
mkdir -p training/datasets/imaterialist

echo ""
echo "📥 Downloading iMaterialist Fashion 2019..."
echo "   This is 1M+ fashion images with fine-grained labels!"
echo "   Size: ~50GB (may take a while)"
echo ""

# Download competition data
kaggle competitions download -c imaterialist-fashion-2019-FGVC6 -p training/datasets/imaterialist

# Check if download succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "📂 Extracting..."
    cd training/datasets/imaterialist
    unzip -q "*.zip"
    echo ""
    echo "✅ iMaterialist dataset ready!"
    echo "   Location: training/datasets/imaterialist/"
    echo ""
    echo "🚀 To train on this data, run:"
    echo "   python3 training/scripts/train_yolo_clothing.py --dataset imaterialist --epochs 100"
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "Alternative: Download manually from:"
    echo "https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data"
fi
