# 👗 AIWardrobe - Computer Vision Partner Overview

## Executive Summary

**AIWardrobe** is a cutting-edge AI-powered fashion technology platform that transforms how users interact with their wardrobe. Our mobile application (React Native/Expo) enables users to **scan their clothing via video**, receive AI-powered outfit recommendations, and virtually try on clothes—all powered by a sophisticated multi-model Computer Vision pipeline.

We're seeking a **Computer Vision specialist** to join as a technical partner/co-founder to help scale our AI capabilities to production-grade accuracy.

---

## 🧠 AI/CV Architecture Overview

### Current Vision Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VIDEO INPUT (Mobile Camera)                      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    📹 SMART FRAME SELECTION                              │
│  • MediaPipe Pose Detection - scores frontal body position              │
│  • Quality Metrics - sharpness, brightness, visibility                  │
│  • Automatic Best Frame Selection                                        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    🎯 MULTI-MODEL DETECTION ENSEMBLE                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  YOLOv8-Seg      │  │  Florence-2      │  │  SegFormer-B2-Clothes│   │
│  │  (Detection +    │  │  (VLM Detection  │  │  (18-Category         │   │
│  │   Segmentation)  │  │   + Grounding)   │  │   Semantic Seg)       │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬──────────┘   │
│           │                     │                         │              │
│           └─────────────────────┴─────────────────────────┘              │
│                                 │                                        │
│                                 ▼                                        │
│                    ┌────────────────────────┐                           │
│                    │   Ensemble Voting +    │                           │
│                    │   Confidence Fusion    │                           │
│                    └────────────────────────┘                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    👕 ATTRIBUTE EXTRACTION                               │
│  • Fashion-CLIP Classification (40+ categories)                         │
│  • Hierarchical Classifier (type → subtype → style)                     │
│  • K-Means Color Clustering (80+ named colors)                          │
│  • Pattern Detection (13 types: solid, striped, floral...)              │
│  • Material Estimation (cotton, denim, leather, silk...)                │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    📸 PRODUCT CARD GENERATION                            │
│  • Alpha-channel mask extraction                                         │
│  • Background removal with edge refinement                               │
│  • Studio lighting normalization                                         │
│  • E-commerce quality output (1000x1000px)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Stack (CV Modules)

### Detection & Segmentation Models

| Model | Purpose | Status |
|-------|---------|--------|
| **YOLOv8-Seg** | Real-time detection + instance segmentation | ✅ Integrated |
| **SegFormer-B2-Clothes** | 18-category semantic segmentation | ✅ Primary |
| **Florence-2** | Vision-Language grounded detection | ✅ Integrated |
| **Grounded SAM2** | Text-prompted segmentation | ✅ Adaptive refinement |
| **MediaPipe Pose** | Body landmark detection for frame scoring | ✅ Integrated |

### Feature Extraction & Classification

| Module | Capability | Status |
|--------|------------|--------|
| **Fashion-CLIP** | Zero-shot fashion classification | ✅ Integrated |
| **Contrastive Fashion Encoder** | Feature embedding for rare items | ✅ Implemented |
| **Fashion Domain Adapter** | CLIP fine-tuned for fashion domain | ✅ Implemented |
| **Hierarchical Classifier** | Multi-level taxonomy (type→subtype→style) | ✅ Implemented |
| **Multi-Scale Feature Pyramid** | Scale-invariant attribute extraction | ✅ Implemented |

### Video Analysis Pipeline

| Component | Function | Status |
|-----------|----------|--------|
| **Temporal Ensemble** | Cross-frame detection aggregation | ✅ Implemented |
| **FeatureSORT Tracker** | Person-anchored outfit tracking | ✅ Implemented |
| **Low-Light Enhancer** | Preprocessing for poor lighting | ✅ Implemented |
| **Slow-Fast Architecture** | Efficient video processing | ✅ Implemented |
| **Outfit Timeline Analyzer** | Multi-outfit detection in videos | ✅ Implemented |

---

## 📊 Current Detection Categories

### SegFormer 18-Category Taxonomy

| # | Category | Examples |
|---|----------|----------|
| 1 | **Upper Clothes** | T-shirts, shirts, blouses, sweaters |
| 2 | **Pants** | Jeans, trousers, chinos, joggers |
| 3 | **Dress** | Maxi, midi, mini, gowns |
| 4 | **Skirt** | A-line, pencil, pleated |
| 5 | **Jacket** | Blazers, bombers, leather, denim |
| 6 | **Coat** | Overcoats, trench, puffer |
| 7 | **Shoes** | Sneakers, boots, heels, sandals |
| 8 | **Bag** | Handbags, backpacks, totes |
| 9 | **Hat** | Caps, beanies, fedoras |
| 10 | **Sunglasses** | All eyewear |
| 11 | **Scarf** | Scarves, shawls |
| 12 | **Belt** | All belt types |
| 13-18 | Body parts | Face, hair, arms, legs (filtered) |

### Fashion-CLIP Extended Classification (40+ types)

**Tops**: T-shirt, Button-down, Polo, Hoodie, Sweater, Cardigan, Tank top, Blouse, Crop top  
**Jackets**: Denim jacket, Leather jacket, Bomber, Blazer, Puffer, Windbreaker  
**Bottoms**: Jeans, Chinos, Cargo pants, Joggers, Shorts, Leggings  
**Shoes**: Running sneakers, High-tops, Chelsea boots, Loafers, Heels, Sandals  

---

## 🎯 Attribute Extraction Capabilities

### Color Analysis
- **80+ Named Colors** with hex values
- **Dominant color extraction** via K-Means clustering
- **Color palette** (primary + secondary with percentages)

### Pattern Detection (13 Types)
`solid` `striped` `plaid` `floral` `polka_dot` `geometric` `animal_print` `paisley` `camo` `tie_dye` `abstract` `checkered` `graphic`

### Material Estimation (10 Types)
`cotton` `denim` `leather` `wool` `silk` `synthetic` `linen` `knit` `fleece` `velvet`

### Detailed Features
- **Collars**: 18 types (crew neck, v-neck, polo, hooded, turtleneck...)
- **Sleeves**: 12 types (short, long, 3/4, sleeveless, puff...)
- **Closures**: 16 types (zip, button-up, pullover, snap, toggle...)
- **Fit**: 9 types (slim, regular, oversized, tailored...)
- **Pockets**: 11 types (side, cargo, chest, patch, no pockets...)

---

## 📈 Performance Metrics

### Current Accuracy

| Metric | Score | Target |
|--------|-------|--------|
| Basic Detection (tops, pants, shoes) | 85% | 98% |
| Fine-grained Classification | 70% | 95% |
| Color Accuracy | 90% | 95% |
| Pattern Detection | 75% | 90% |
| False Positive Rate | 5% | <2% |

### Processing Performance

| Operation | Time | Hardware |
|-----------|------|----------|
| Frame Selection | ~50ms | CPU |
| YOLOv8 Detection | ~100ms | GPU |
| SegFormer Inference | ~800ms | GPU |
| Attribute Extraction | ~250ms | CPU/GPU |
| Product Card Generation | ~50ms | CPU |
| **Total Pipeline** | **~1.5s** | Mixed |

---

## 🚀 Technical Opportunities for CV Partner

### Immediate Priorities

1. **Fine-grained Classification Accuracy**
   - Distinguish jacket types (denim vs leather vs bomber)
   - Improve footwear classification (sneakers vs boots vs sandals)
   - Better pattern recognition for complex designs

2. **Video Analysis Robustness**
   - Improve tracking across outfit changes
   - Handle motion blur and variable lighting
   - Optimize for real-time mobile processing

3. **Edge Cases & Rare Items**
   - Designer/luxury item recognition
   - Cultural/regional clothing types
   - Accessories with low visibility

### Advanced R&D Opportunities

1. **Grounded SAM2 Expansion**
   - Text-prompted fine-grained detection
   - "Find the denim jacket with chest pockets"
   - User-guided refinement

2. **Multi-Modal Fusion**
   - Combine visual + text descriptions
   - Brand/logo detection
   - Style similarity search

3. **3D Garment Understanding**
   - Gaussian splatting for 3D reconstruction
   - Virtual try-on improvements
   - Size estimation from video

4. **Active Learning Pipeline**
   - User feedback integration
   - Continuous model improvement
   - Domain-specific fine-tuning

---

## 🏗️ Codebase Overview

### Repository Structure

```
AIWardrobe/
├── alicevision-service/          # Python CV Backend (FastAPI)
│   ├── main.py                   # API endpoints (270KB+)
│   ├── modules/                  # 78 CV modules
│   │   ├── segmentation.py       # SegFormer integration
│   │   ├── ensemble_detector.py  # Multi-model fusion
│   │   ├── fashion_clip.py       # Fashion-CLIP classification
│   │   ├── hierarchical_classifier.py
│   │   ├── temporal_ensemble.py  # Video analysis
│   │   ├── grounded_sam.py       # Text-prompted detection
│   │   ├── low_light_enhancer.py
│   │   ├── attribute_extractor.py
│   │   └── ... (70+ more modules)
│   ├── sam2/                     # Segment Anything 2
│   ├── groundingdino/            # Grounding DINO
│   └── weights/                  # YOLOv8 models
├── api/                          # Node.js middleware
├── screens/                      # React Native frontend
└── components/                   # UI components
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Mobile** | React Native, Expo, TypeScript |
| **CV Backend** | Python 3.10+, FastAPI, PyTorch |
| **CV Models** | YOLOv8, SegFormer, SAM2, CLIP, Florence-2 |
| **Infrastructure** | Docker, Render/Vercel deployment |

---

## 🤝 What We're Looking For

### Ideal Partner Profile

- **Strong CV/ML background** (Master's/PhD or equivalent experience)
- **PyTorch proficiency** with production deployment experience
- **Familiarity with**:
  - Object detection (YOLO family, DETR, etc.)
  - Semantic/Instance segmentation (Mask R-CNN, SAM, SegFormer)
  - Vision transformers (ViT, CLIP, Florence)
  - Video understanding and tracking
- **Bonus**: Fashion/retail domain experience

### Partnership Structure

- **Equity-based partnership** (startup stage, pre-seed)
- **Technical co-founder role** with ownership stake
- **Full autonomy** over CV architecture decisions
- **Remote-friendly** collaboration

---

## 📬 Contact

**Founder**: Zohid Vohidjonov  
**GitHub**: [@abduzahid8](https://github.com/abduzahid8)  
**Project**: [AIWardrobe Repository](https://github.com/abduzahid8/AIWardrobe)

---

*Built with ❤️ and Computer Vision*
