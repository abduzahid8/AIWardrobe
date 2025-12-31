# CVAT Annotation Setup for AIWardrobe

## Why CVAT?
CVAT (Computer Vision Annotation Tool) is the best free option for clothing annotation because:
- ✅ Open source and free
- ✅ Polygon segmentation (pixel-perfect masks)
- ✅ Automatic annotation with AI models
- ✅ Team collaboration
- ✅ YOLO/COCO export formats

## Quick Setup (Docker)

```bash
# Clone CVAT
git clone https://github.com/opencv/cvat
cd cvat

# Start with Docker
docker-compose up -d

# Access at: http://localhost:8080
```

## Cloud Option (Free)
Use CVAT.ai cloud: https://app.cvat.ai
- Free tier: 10 concurrent tasks
- No setup required

## Annotation Workflow

### 1. Create Project
- Name: "AIWardrobe Clothing"
- Labels: (see list below)

### 2. Create Labels (46 categories)
```
Tops: t-shirt, shirt, polo, sweater, hoodie, cardigan, tank_top, blouse
Outerwear: jacket, blazer, coat, parka, bomber, leather_jacket, denim_jacket
Bottoms: jeans, pants, trousers, chinos, joggers, shorts, skirt, leggings
Dresses: dress, maxi_dress, midi_dress, mini_dress
Footwear: sneakers, boots, loafers, heels, sandals, flats
Accessories: bag, backpack, handbag, hat, cap, belt, scarf, sunglasses
```

### 3. Annotation Types
- **Bounding Box**: Fast, for detection only
- **Polygon**: Best accuracy, for segmentation
- **Mask**: Pixel-perfect, slowest

### 4. Export Format
- YOLO 1.1 (for YOLOv8 training)
- COCO 1.0 (for SegFormer/Mask R-CNN)

## AI-Assisted Annotation
CVAT supports automatic annotation:
1. Go to Models → Add Model
2. Use pretrained detector for initial boxes
3. Manually refine

## Keyboard Shortcuts
- `N`: New bounding box
- `Shift+N`: New polygon
- `F`: Finish polygon
- `D`: Delete selected
- `Ctrl+S`: Save
