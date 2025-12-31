# AIWardrobe AI Training 🎨

Train your own clothing detection AI using **2M+ free images**.

## Quick Start (FREE - Google Colab)

1. **Open the Colab notebook:**
   [Train_AIWardrobe_Clothing_AI.ipynb](notebooks/Train_AIWardrobe_Clothing_AI.ipynb)

2. **Upload to Google Drive**

3. **Click "Run All"**
   - Uses free T4 GPU
   - ~4 hours training time
   - 46 clothing categories

4. **Download trained model** (`best.pt`)

5. **Use in AIWardrobe:**
   ```bash
   cp best.pt ../weights/clothing_detector.pt
   ```

## Datasets (2M+ Images FREE)

| Dataset | Images | Categories | Download |
|---------|--------|------------|----------|
| iMaterialist | 1M+ | 46 | `python download_datasets.py --dataset imaterialist` |
| DeepFashion2 | 800K | 13 | Manual (Google Drive) |
| ModaNet | 55K | 13 | `python download_datasets.py --dataset modanet` |
| Fashion-MNIST | 70K | 10 | `python download_datasets.py --dataset fashion_mnist` |

## Annotation (CVAT)

See [CVAT_ANNOTATION_GUIDE.md](CVAT_ANNOTATION_GUIDE.md)

Free cloud: https://app.cvat.ai

## 46 Clothing Categories

```
Tops: t-shirt, shirt, polo, sweater, hoodie, cardigan, tank_top, blouse
Outerwear: jacket, blazer, coat, parka, bomber, leather_jacket, denim_jacket
Bottoms: jeans, pants, trousers, chinos, joggers, shorts, skirt, leggings
Dresses: dress, maxi_dress, midi_dress, mini_dress
Footwear: sneakers, boots, loafers, heels, sandals, flats
Accessories: bag, backpack, hat, cap, belt, scarf, sunglasses
```

## Files

```
training/
├── notebooks/
│   └── Train_AIWardrobe_Clothing_AI.ipynb  # Colab notebook
├── scripts/
│   ├── train_yolo_clothing.py              # Local training
│   └── download_datasets.py                # Dataset downloader
├── datasets/                               # Downloaded data
└── CVAT_ANNOTATION_GUIDE.md               # Annotation guide
```
