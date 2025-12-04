# Clarifai Video Wardrobe Analysis - Setup Guide

## 🚀 Quick Start

### 1. Get Clarifai API Key

1. Go to https://clarifai.com/signup
2. Create a free account
3. Go to Settings → Security → Create Personal Access Token
4. Copy your API key

### 2. Backend Setup

#### Install Dependencies
```bash
cd api
npm install multer fluent-ffmpeg axios
```

#### Add to your Express server (api/index.js or api/server.js)
```javascript
const wardrobeAnalysisRoutes = require('./routes/wardrobeAnalysis');

// Add this line with your other routes
app.use('/api', wardrobeAnalysisRoutes);
```

#### Set Environment Variable
Create or update `api/.env`:
```
CLARIFAI_API_KEY=your_actual_api_key_here
```

#### Install FFmpeg (Required for video processing)

**Mac:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 3. Frontend Setup

#### Install Dependencies
```bash
npm install expo-image-picker expo-document-picker
```

#### Add to Navigation

Update `navigation/RootNavigator.tsx`:
```typescript
import WardrobeVideoScreen from '../screens/WardrobeVideoScreen';

// Add this screen to your stack
<Stack.Screen 
  name="WardrobeVideo" 
  component={WardrobeVideoScreen}
  options={{ title: 'Scan Wardrobe' }}
/>
```

#### Add Navigation Button

In your `DiscoverScreen` or `ProfileScreen`, add a button:
```typescript
<TouchableOpacity 
  onPress={() => navigation.navigate('WardrobeVideo')}
  style={styles.scanButton}
>
  <Ionicons name="videocam" size={24} color="#FFF" />
  <Text>Scan Wardrobe</Text>
</TouchableOpacity>
```

### 4. Test the Feature

1. Start your backend:
```bash
cd api
npm start
```

2. Start your React Native app:
```bash
npx expo start
```

3. Navigate to the Wardrobe Video screen
4. Upload a video of your wardrobe
5. Wait for analysis (30-90 seconds)
6. Review detected items!

## 📊 How It Works

1. **User uploads video** → Frontend sends to backend
2. **Backend extracts frames** → FFmpeg extracts 1 frame/second
3. **Clarifai analyzes frames** → Detects clothing in each frame
4. **Results aggregated** → Deduplicates and ranks by confidence
5. **User reviews items** → Can save to digital wardrobe

## 💰 Pricing

**Clarifai Free Tier:**
- 1,000 operations/month FREE
- Perfect for testing and small user base

**After Free Tier:**
- $1.20 per 1,000 operations
- 1 video (60 seconds) = ~60 operations
- So ~16 videos per dollar

## 🔧 Troubleshooting

### "FFmpeg not found"
- Install FFmpeg (see instructions above)
- Restart your terminal/server

### "Clarifai API error"
- Check your API key is correct in `.env`
- Verify you have operations remaining in your Clarifai account

### "Video upload fails"
- Check video file size (max 100MB)
- Try shorter video (1-2 minutes recommended)

### "No items detected"
- Ensure video shows clothing clearly
- Try better lighting
- Make sure clothes are visible (not in drawers/boxes)

## 📱 Best Practices for Users

**For Best Results:**
- ✅ Record in good lighting
- ✅ Show each item for 2-3 seconds
- ✅ Keep camera steady
- ✅ Film from 1-2 feet away
- ✅ Videos 30-120 seconds work best

**Avoid:**
- ❌ Very dark videos
- ❌ Shaky/blurry footage
- ❌ Items too far away
- ❌ Videos longer than 5 minutes

## 🎯 Next Steps

After implementation, you can:
1. Save detected items to user's database
2. Allow users to edit/confirm items
3. Add color detection
4. Implement outfit suggestions based on detected items

## 📞 Support

If you need help:
1. Check Clarifai docs: https://docs.clarifai.com
2. Check FFmpeg docs: https://ffmpeg.org/documentation.html
3. Review error logs in backend console
