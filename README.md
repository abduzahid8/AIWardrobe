# 👗 AIWardrobe

A smart AI-powered wardrobe management app built with React Native and Expo. Scan your wardrobe with video, get outfit suggestions, and try on clothes virtually!

![React Native](https://img.shields.io/badge/React_Native-0.81.5-blue)
![Expo](https://img.shields.io/badge/Expo-54-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue)

## ✨ Features

- 🎥 **Video Wardrobe Scanning** - Scan your clothes using your phone camera
- 🤖 **AI Clothing Detection** - Automatically identify and categorize clothing items
- 👔 **Virtual Try-On** - See how clothes look on you using AI
- 🎨 **Product Photo Generation** - Get professional e-commerce style photos of your items
- 💬 **AI Fashion Assistant** - Chat with an AI stylist for outfit advice
- 🌤️ **Weather-Based Suggestions** - Get outfit recommendations based on weather
- 🌍 **Multi-Language Support** - Available in English, Russian, and Uzbek

## 📱 Screenshots

<!-- Add your screenshots here -->

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Expo CLI: `npm install -g expo-cli`
- iOS Simulator (Mac) or Android Emulator, or Expo Go app on your phone

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AIWardrobe.git
   cd AIWardrobe
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the `/api` directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   REPLICATE_API_TOKEN=your_replicate_token
   HF_TOKEN=your_huggingface_token
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

4. **Start the API server**
   ```bash
   cd api
   npm install
   node index.js
   ```

5. **Start the Expo app**
   ```bash
   # In the root directory
   npm start
   ```

6. **Open on your device**
   - Scan the QR code with Expo Go (Android) or Camera app (iOS)
   - Or press `i` for iOS Simulator / `a` for Android Emulator

## 📁 Project Structure

```
AIWardrobe/
├── api/                      # Express.js backend
│   ├── routes/               # Modular API routes
│   │   ├── auth.js           # Authentication (register, login)
│   │   ├── ai.js             # AI endpoints (analyze, generate)
│   │   ├── clothing.js       # Clothing CRUD
│   │   ├── outfits.js        # Outfit management
│   │   └── weather.js        # Weather API
│   ├── middleware/           # Express middleware
│   │   └── auth.js           # JWT authentication
│   └── models/               # MongoDB models
├── components/               # React components
│   ├── ui/                   # Reusable UI components
│   │   ├── Card.tsx          # Card with variants
│   │   ├── Input.tsx         # Styled input
│   │   ├── Header.tsx        # Screen header
│   │   ├── EmptyState.tsx    # Empty placeholders
│   │   ├── LoadingState.tsx  # Skeleton loaders
│   │   └── Toast.tsx         # Notifications
│   ├── AnimatedButton.tsx    # Animated button
│   ├── LanguageSelector.tsx  # Language picker
│   └── StyleSelector.tsx     # Designer style picker
├── screens/                  # Screen components
├── navigation/               # React Navigation setup
├── src/
│   ├── types/                # TypeScript type definitions
│   └── theme/                # Theme configuration
├── i18n/                     # Internationalization files
└── store/                    # Zustand state management
```

## 🛠️ Available Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start Expo development server |
| `npm run ios` | Run on iOS Simulator |
| `npm run android` | Run on Android Emulator |
| `npm run web` | Run in browser |
| `npm run lint` | Check code for issues |
| `npm run lint:fix` | Auto-fix lint issues |
| `npm run format` | Format code with Prettier |
| `npm run typecheck` | Check TypeScript types |

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Register new user |
| POST | `/login` | Login user |
| GET | `/me` | Get current user |

### Clothing
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/clothing-items` | Get user's clothes |
| POST | `/clothing-items` | Add clothing item |
| POST | `/wardrobe/add-batch` | Bulk add items |

### AI Features
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze-frames` | Analyze video frames |
| POST | `/scan-wardrobe` | Scan wardrobe from video |
| POST | `/try-on` | Virtual try-on |
| POST | `/ai-chat` | Chat with AI stylist |
| GET | `/smart-search` | AI-powered search |

### Weather
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/weather` | Get weather by city |
| POST | `/weather/coords` | Get weather by coordinates |

## 🧩 Tech Stack

### Frontend
- **React Native** - Cross-platform mobile framework
- **Expo** - Development platform
- **TypeScript** - Type safety
- **React Navigation** - Navigation library
- **Zustand** - State management
- **React Native Reanimated** - Animations
- **NativeWind** - Tailwind CSS for React Native

### Backend
- **Express.js** - Node.js web framework
- **MongoDB** - Database
- **Supabase** - File storage
- **JWT** - Authentication

### AI Services
- **Google Gemini** - Video/image analysis
- **OpenAI GPT-4** - Clothing analysis
- **Replicate** - Image generation & virtual try-on
- **Hugging Face** - Embeddings & chat

---

## 🧠 AI Vision System

AIWardrobe uses a powerful, multi-stage AI pipeline for intelligent clothing analysis. Here's how it works:

### How Our AI Works

#### 1. 📹 Smart Frame Selection
When you scan your wardrobe with video, our AI analyzes every frame to find the best one:

- **MediaPipe Pose Detection** - Scores each frame for frontal body position
- **Quality Metrics** - Evaluates sharpness, brightness, and object visibility
- **Automatic Selection** - Picks the clearest frame with the best clothing visibility

#### 2. 👕 18-Category Clothing Segmentation
Powered by **SegFormer-B2-Clothes** (Hugging Face transformer model):

| Category | Examples |
|----------|----------|
| **Upper Clothes** | Shirts, blouses, sweaters, hoodies, t-shirts, cardigans |
| **Pants** | Jeans, trousers, chinos, joggers, leggings, cargo pants |
| **Dress** | Maxi, midi, mini dresses, gowns, sundresses |
| **Skirt** | Mini, maxi, midi, pleated, pencil skirts |
| **Jacket** | Blazers, coats, bombers, leather jackets, puffers |
| **Shoes** | Sneakers, loafers, boots, heels, sandals, oxfords |
| **Bag** | Handbags, backpacks, totes, clutches, crossbodies |
| **Hat** | Caps, beanies, fedoras, bucket hats |
| **Sunglasses** | All eyewear types |
| **Scarf** | Scarves, wraps, shawls |
| **Belt** | All belt types |

The AI creates pixel-perfect masks for each item, enabling:
- ✅ Individual item cutouts
- ✅ White background removal
- ✅ Professional product card generation

#### 3. 🎨 Color & Pattern Analysis
Our AI extracts detailed color information using K-means clustering:

- **80+ Named Colors** - From "Navy Blue" to "Dusty Rose" to "Cognac"
- **Color Palette Extraction** - Primary + secondary colors with percentages
- **Pattern Detection** - Solid, striped, plaid, floral, geometric, animal print
- **Material Hints** - Cotton, denim, leather, wool, silk, synthetic

#### 4. 📸 Professional Product Cards
After segmentation, we create e-commerce quality images:

- **Clean Background Removal** - Alpha channel transparency
- **Edge Refinement** - Smooth, anti-aliased cutouts
- **Studio Lighting Normalization** - Consistent brightness & color temperature
- **4 Templates** - Catalog, Minimal, Lifestyle, E-commerce styles

### Complete AI Endpoints

#### AliceVision Service (Port 5050)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment-all` | POST | Detect & cut out ALL clothing items in an image |
| `/segment` | POST | Full image segmentation with combined mask |
| `/segment-item` | POST | Cut out specific item by bounding box |
| `/keyframe` | POST | Select best frame from video frames |
| `/pose` | POST | Analyze body pose quality in frames |
| `/lighting` | POST | Studio-quality lighting normalization |
| `/card` | POST | Generate professional product card |
| `/process` | POST | Complete video → product card pipeline |
| `/analyze-product` | POST | Full YOLOv8 + Fashion-CLIP analysis |
| `/extract-attributes` | POST | Extract colors, patterns, materials |
| `/assess-quality` | POST | E-commerce photo quality scoring |
| `/find-similar` | POST | Visual similarity search |

#### Example: `/segment-all` Response
```json
{
  "success": true,
  "totalItems": 4,
  "items": [
    {
      "category": "upper_clothes",
      "primaryColor": "Navy Blue",
      "colorHex": "#1B3A57",
      "confidence": 0.94,
      "bbox": [120, 80, 280, 320],
      "cutoutImage": "data:image/png;base64,..."
    },
    {
      "category": "pants",
      "primaryColor": "Olive",
      "colorHex": "#6B7B3C",
      "confidence": 0.91,
      "bbox": [100, 340, 300, 640],
      "cutoutImage": "data:image/png;base64,..."
    }
  ],
  "processingTimeMs": 1250.5
}
```

### AI Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    📱 Mobile App (React Native)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 🔌 Node.js API Server (Port 3000)            │
│  • Routes: /scan-wardrobe, /try-on, /ai-chat                │
│  • Integrations: OpenAI, Gemini, Replicate, Supabase        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              🤖 AliceVision AI Service (Port 5050)           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SegFormer-B2-Clothes ────────────────────────────────│   │
│  │   • 18-category semantic segmentation                │   │
│  │   • Per-item masks with confidence scores            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MediaPipe Pose ──────────────────────────────────────│   │
│  │   • Body landmark detection                          │   │
│  │   • Frontal pose scoring                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Color Analysis (K-Means) ────────────────────────────│   │
│  │   • 80+ color names                                  │   │
│  │   • Dominant color extraction                        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Product Card Generator ──────────────────────────────│   │
│  │   • Edge refinement & alpha masking                  │   │
│  │   • 4 professional templates                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Running the AI Service Locally

```bash
# Start AliceVision service
cd alicevision-service
pip install -r requirements.txt
python main.py  # Runs on port 5050

# Start Node.js API
cd api
npm install
node index.js   # Runs on port 3000
```


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Zohid Vohidjonov**

- GitHub: [@abduzahid8](https://github.com/abduzahid8)

---

Made with ❤️ and AI
