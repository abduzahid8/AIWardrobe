# New AI Endpoints - API Documentation

## Overview
Two new powerful endpoints have been added to provide comprehensive clothing analysis beyond basic detection.

---

## 1. `/analyze-attributes` - Comprehensive Attribute Analysis

### Description
Analyzes a clothing image and returns detailed attributes including colors, patterns, materials, and texture information.

### Request
```json
POST /analyze-attributes
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Response
```json
{
  "success": true,
  "attributes": {
    "colors": [
      {
        "name": "dark blue",
        "rgb": [25, 51, 102],
        "hex": "#193366",
        "percentage": 65.4
      },
      {
        "name": "white",
        "rgb": [255, 255, 255],
        "hex": "#FFFFFF",
        "percentage": 22.1
      }
    ],
    "primaryColor": "dark blue",
    "colorPalette": ["#193366", "#FFFFFF", "#C0C0C0"],
    "pattern": {
      "type": "striped",
      "confidence": 0.75,
      "complexity": 0.45,
      "description": "Moderate striped pattern"
    },
    "material": {
      "type": "cotton",
      "confidence": 0.75,
      "texture": "woven"
    },
    "processingTimeMs": 245.3
  },
  "processingTimeMs": 245.3
}
```

### Use Cases
- **Color Matching**: Find items with similar color palettes
- **Pattern Search**: Search for striped, solid, or floral items
- **Material Filtering**: Filter by cotton, denim, silk, etc.
- **Style Recommendations**: Use attributes for outfit recommendations

---

## 2. `/classify-style` - Fashion-CLIP Style Classification

### Description
Uses Fashion-CLIP AI to accurately classify clothing type and extract style tags. More accurate than SegFormer for specific clothing categorization.

### Request
```json
POST /classify-style
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "include_subcategory": true
}
```

### Response
```json
{
  "success": true,
  "category": "denim jacket",
  "confidence": 0.89,
  "subcategory": "trucker jacket",
  "styleTags": ["casual", "vintage", "classic"],
  "processingTimeMs": 892.1
}
```

### Supported Categories
#### Tops
- t-shirt, shirt, blouse, sweater, hoodie, jacket, coat, blazer, cardigan, vest, polo shirt

#### Bottoms
- pants, jeans, shorts, leggings, skirt

#### Dresses & One-Pieces
- dress, suit

#### Footwear
- sneakers, boots, sandals, heels, flats

#### Accessories
- hat, cap, beanie, scarf, gloves, bag, backpack, purse, handbag, sunglasses, glasses, watch, belt, tie

#### Athletic & Special
- athletic wear, swimwear

### Style Tags
casual, formal, business, sporty, athletic, elegant, vintage, modern, streetwear, bohemian, minimalist, luxury, comfortable, trendy, classic

### Use Cases
- **Accurate Classification**: Better than SegFormer for specific types like "denim jacket" vs just "jacket"
- **Style-Based Search**: Find items with specific style tags
- **Outfit Coordination**: Match items by style compatibility
- **Trend Analysis**: Track which styles are most popular

---

## 3. Enhanced `/segment-all` Response

### What Changed
Each detected item now includes a full `attributes` object with detailed analysis.

### Previous Response
```json
{
  "category": "Top",
  "primaryColor": "Dark Blue",
  "colorHex": "#193366",
  "confidence": 0.94,
  "bbox": [120, 80, 350, 420],
  "cutoutImage": "data:image/png;base64,..."
}
```

### New Response (WITH Attributes)
```json
{
  "category": "Top",
  "primaryColor": "Dark Blue",
  "colorHex": "#193366",
  "confidence": 0.94,
  "bbox": [120, 80, 350, 420],
  "cutoutImage": "data:image/png;base64,...",
  "attributes": {
    "colors": [
      {"name": "dark blue", "hex": "#193366", "percentage": 65.4},
      {"name": "white", "hex": "#FFFFFF", "percentage": 22.1}
    ],
    "pattern": {
      "type": "striped",
      "confidence": 0.75,
      "description": "Moderate striped pattern"
    },
    "material": {
      "type": "cotton",
      "confidence": 0.75,
      "texture": "woven"
    }
  }
}
```

---

## Complete Clothing Attribute Categories

### Pattern Types (13)
solid, striped, plaid, checkered, polka dot, floral, geometric, animal print, camouflage, tie-dye, gradient, abstract, printed text

### Material Types (10)
cotton, silk, denim, leather, wool, polyester, linen, velvet, satin, knit

### Collar/Neckline Types (18)
crew neck, v-neck, scoop neck, turtleneck, mock neck, polo collar, button-down collar, spread collar, mandarin collar, hooded, cowl neck, boat neck, square neck, halter, off-shoulder, one-shoulder, strapless, peter pan collar

### Sleeve Types (12)
short sleeve, long sleeve, 3/4 sleeve, sleeveless, cap sleeve, raglan sleeve, puff sleeve, bell sleeve, bishop sleeve, dolman sleeve, roll-up sleeve, cuffed

### Closure Types (16)
pullover, full zip, half zip, quarter zip, button-up, button-down, snap buttons, toggle buttons, drawstring, hook and eye, velcro, wrap, tie front, hidden placket, double-breasted, single-breasted

### Fit Types (9)
slim fit, regular fit, relaxed fit, oversized, tailored, loose, fitted, cropped, boxy

### Length Types (9)
cropped, waist-length, hip-length, thigh-length, knee-length, midi, maxi, full-length, ankle-length

### Pocket Types (11)
no pockets, side pockets, front pockets, back pockets, kangaroo pocket, chest pocket, flap pockets, zippered pockets, patch pockets, welt pockets, cargo pockets

---

## Example: Complete Detection Flow

### Step 1: Detect All Items
```bash
POST /segment-all
```
Returns: 3 items (Top, Pants, Hat) with cutouts and attributes

### Step 2 (Optional): Detailed Analysis
For each item, optionally call:
```bash
POST /analyze-attributes  # Get full color analysis
POST /classify-style      # Get precise clothing type
```

### Step 3: Use the Data
- Display attributes in UI
- Enable filtering by pattern/material
- Create outfit recommendations
- Match items by color palette

---

## Performance Notes

| Endpoint | Avg Time | Notes |
|----------|----------|-------|
| `/segment-all` | ~1500ms | Includes attribute extraction |
| `/analyze-attributes` | ~250ms | Fast color/pattern analysis |
| `/classify-style` | ~900ms | Fashion-CLIP model inference |

---

## Testing the Endpoints

### Test `/analyze-attributes`
```bash
curl -X POST http://localhost:5050/analyze-attributes \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64_encoded_image>"}'
```

### Test `/classify-style`
```bash
curl -X POST http://localhost:5050/classify-style \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64_encoded_image>","include_subcategory":true}'
```

---

## Integration Example (React Native)

```typescript
// Detect all items with attributes
const response = await axios.post(`${ALICEVISION_URL}/segment-all`, {
  image: base64Image,
  add_white_background: true
});

response.data.items.forEach(item => {
  console.log(`${item.category}:`);
  console.log(`  Color: ${item.primaryColor}`);
  
  if (item.attributes) {
    console.log(`  Pattern: ${item.attributes.pattern.type}`);
    console.log(`  Material: ${item.attributes.material.type}`);
    console.log(`  Color Palette: ${item.attributes.colorPalette.join(', ')}`);
  }
});
```

---

## Future Enhancements

### Potential Additions
- [ ] Zipper detection (full zip, half zip, no zip)
- [ ] Button count and style
- [ ] Specific collar type detection
- [ ] Sleeve length measurement
- [ ] Embellishment detection (sequins, embroidery, etc.)
- [ ] Condition assessment (new, worn, vintage)
- [ ] Brand logo detection
- [ ] Size estimation from measurements
