# Clarifai Video Wardrobe Analysis - Complete Implementation

## ✅ What's Been Created

### Backend Files
1. **`api/routes/wardrobeAnalysis.js`** - Complete API route for video analysis
2. **`api/package.json`** - Updated with `fluent-ffmpeg` dependency

### Frontend Files
1. **`screens/WardrobeVideoScreen.tsx`** - Full React Native screen
2. **`navigation/RootNavigator.tsx`** - Updated with WardrobeVideo route

### Documentation
1. **`CLARIFAI_SETUP_GUIDE.md`** - Complete setup instructions

## 🚀 Installation Steps

### Step 1: Install Backend Dependencies
```bash
cd api
npm install
```

This will install the new `fluent-ffmpeg` package.

### Step 2: Install FFmpeg
**Mac:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

### Step 3: Get Clarifai API Key
1. Sign up at https://clarifai.com/signup
2. Go to Settings → Security → Create Personal Access Token
3. Copy your API key

### Step 4: Configure Environment
Create or update `api/.env`:
```
CLARIFAI_API_KEY=your_actual_api_key_here
```

### Step 5: Add Route to Express Server
In `api/index.js` or `api/server.js`, add:
```javascript
const wardrobeAnalysisRoutes = require('./routes/wardrobeAnalysis');
app.use('/api', wardrobeAnalysisRoutes);
```

### Step 6: Install Frontend Dependencies
```bash
npm install expo-image-picker
```

### Step 7: Test!
1. Start backend: `cd api && npm start`
2. Start frontend: `npx expo start`
3. Navigate to Profile → Scan Wardrobe (you'll need to add a button)

## 📱 Adding Navigation Button

Add this to your `ProfileScreen` or `DiscoverScreen`:

```tsx
<TouchableOpacity 
  style={styles.scanButton}
  onPress={() => (navigation as any).navigate('WardrobeVideo')}
>
  <Ionicons name="videocam" size={24} color="#FFF" />
  <Text style={styles.scanButtonText}>Scan Wardrobe</Text>
</TouchableOpacity>
```

## 🎯 How It Works

1. User uploads wardrobe video
2. Backend extracts 1 frame per second using FFmpeg
3. Each frame is sent to Clarifai Fashion API
4. AI detects clothing items (shirts, pants, shoes, etc.)
5. Results are aggregated and deduplicated
6. User sees list of detected items with confidence scores

## 💰 Cost Estimate

**Clarifai Pricing:**
- Free: 1,000 operations/month
- Paid: $1.20 per 1,000 operations

**Example:**
- 60-second video = ~60 frames = ~60 operations
- Free tier = ~16 videos/month
- After free tier = ~16 videos per $1

## 📊 What Gets Detected

Clarifai's Fashion API can detect:
- Shirts, T-shirts, Blouses
- Pants, Jeans, Trousers
- Dresses, Skirts
- Jackets, Coats
- Shoes, Boots, Sneakers
- Accessories (bags, hats, etc.)

## 🔧 Troubleshooting

**"FFmpeg not found"**
- Install FFmpeg (see Step 2)
- Restart terminal and server

**"Clarifai API error"**
- Check API key in `.env`
- Verify free tier not exhausted

**"No items detected"**
- Ensure good lighting in video
- Show items clearly for 2-3 seconds each
- Keep camera steady

## 📝 Next Steps

After testing, you can:
1. Save detected items to MongoDB
2. Allow users to edit/confirm items
3. Add color detection
4. Generate outfit suggestions

## 🎉 You're Done!

All code is ready to use. Just follow the installation steps above!
