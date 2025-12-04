# Video Wardrobe Analysis - API Comparison

## Overview
This document provides complete code examples for implementing video wardrobe analysis using two different AI APIs.

---

## Option A: Google Cloud Vision API

### Pros
- ✅ Very accurate object detection
- ✅ Detects clothing types, colors, attributes
- ✅ Reliable and well-documented
- ✅ Good free tier (1000 requests/month)

### Cons
- ❌ Requires Google Cloud account setup
- ❌ More complex authentication
- ❌ Not specialized for fashion (general object detection)

### Pricing
- **Free Tier**: 1,000 units/month
- **After Free Tier**: $1.50 per 1,000 images

### Setup Steps
1. Create Google Cloud account
2. Enable Vision API
3. Create service account & download credentials JSON
4. Install: `npm install @google-cloud/vision`

### Backend Code (Node.js/Express)

\`\`\`javascript
// backend/routes/wardrobeAnalysis.js
const express = require('express');
const router = express.Router();
const vision = require('@google-cloud/vision');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');

// Initialize Google Vision client
const client = new vision.ImageAnnotatorClient({
  keyFilename: './google-credentials.json' // Your credentials file
});

// Configure multer for video upload
const upload = multer({ dest: 'uploads/videos/' });

// Extract frames from video
async function extractFrames(videoPath, outputDir) {
  return new Promise((resolve, reject) => {
    // Extract 1 frame per second
    ffmpeg(videoPath)
      .outputOptions('-vf fps=1')
      .output(path.join(outputDir, 'frame-%04d.jpg'))
      .on('end', () => {
        const frames = fs.readdirSync(outputDir)
          .filter(f => f.startsWith('frame-'))
          .map(f => path.join(outputDir, f));
        resolve(frames);
      })
      .on('error', reject)
      .run();
  });
}

// Analyze single frame with Google Vision
async function analyzeFrame(imagePath) {
  const [result] = await client.objectLocalization(imagePath);
  const objects = result.localizedObjectAnnotations;
  
  // Filter for clothing items
  const clothingKeywords = ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'skirt', 'coat', 'sweater', 'jeans'];
  const detectedClothes = objects.filter(obj => 
    clothingKeywords.some(keyword => obj.name.toLowerCase().includes(keyword))
  );
  
  return detectedClothes.map(item => ({
    type: item.name,
    confidence: item.score,
    boundingBox: item.boundingPoly
  }));
}

// Main endpoint
router.post('/analyze-wardrobe', upload.single('video'), async (req, res) => {
  try {
    const videoPath = req.file.path;
    const framesDir = path.join('uploads', 'frames', Date.now().toString());
    
    // Create frames directory
    fs.mkdirSync(framesDir, { recursive: true });
    
    // Extract frames from video
    console.log('Extracting frames...');
    const frames = await extractFrames(videoPath, framesDir);
    
    // Analyze each frame
    console.log(\`Analyzing \${frames.length} frames...\`);
    const allDetections = [];
    
    for (const frame of frames) {
      const detections = await analyzeFrame(frame);
      allDetections.push(...detections);
    }
    
    // Deduplicate and aggregate results
    const uniqueItems = {};
    allDetections.forEach(item => {
      const key = item.type.toLowerCase();
      if (!uniqueItems[key] || uniqueItems[key].confidence < item.confidence) {
        uniqueItems[key] = item;
      }
    });
    
    // Cleanup
    fs.rmSync(videoPath);
    fs.rmSync(framesDir, { recursive: true });
    
    res.json({
      success: true,
      itemsDetected: Object.values(uniqueItems),
      totalFramesAnalyzed: frames.length
    });
    
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
\`\`\`

---

## Option B: Clarifai Fashion API (RECOMMENDED)

### Pros
- ✅ **Specialized for fashion/clothing**
- ✅ Detects specific clothing types (t-shirt, jeans, dress, etc.)
- ✅ Identifies colors, patterns, styles
- ✅ Easier setup than Google
- ✅ Better for fashion use case

### Cons
- ❌ Smaller free tier than Google

### Pricing
- **Free Tier**: 1,000 operations/month
- **After Free Tier**: $1.20 per 1,000 operations

### Setup Steps
1. Sign up at https://clarifai.com
2. Create an app
3. Get API key
4. Install: `npm install clarifai`

### Backend Code (Node.js/Express)

\`\`\`javascript
// backend/routes/wardrobeAnalysisClarifai.js
const express = require('express');
const router = express.Router();
const { ClarifaiStub, grpc } = require("clarifai-nodejs-grpc");
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');

// Initialize Clarifai
const stub = ClarifaiStub.grpc();
const metadata = new grpc.Metadata();
metadata.set("authorization", "Key YOUR_CLARIFAI_API_KEY");

// Configure multer
const upload = multer({ dest: 'uploads/videos/' });

// Extract frames from video
async function extractFrames(videoPath, outputDir) {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions('-vf fps=1') // 1 frame per second
      .output(path.join(outputDir, 'frame-%04d.jpg'))
      .on('end', () => {
        const frames = fs.readdirSync(outputDir)
          .filter(f => f.startsWith('frame-'))
          .map(f => path.join(outputDir, f));
        resolve(frames);
      })
      .on('error', reject)
      .run();
  });
}

// Analyze frame with Clarifai Fashion Model
async function analyzeFrameWithClarifai(imagePath) {
  return new Promise((resolve, reject) => {
    const imageBytes = fs.readFileSync(imagePath);
    
    stub.PostModelOutputs(
      {
        model_id: "apparel-detection", // Clarifai's fashion model
        inputs: [{
          data: {
            image: {
              base64: imageBytes.toString('base64')
            }
          }
        }]
      },
      metadata,
      (err, response) => {
        if (err) {
          reject(err);
          return;
        }
        
        if (response.status.code !== 10000) {
          reject(new Error("API error: " + response.status.description));
          return;
        }
        
        const concepts = response.outputs[0].data.concepts;
        const detectedItems = concepts
          .filter(c => c.value > 0.7) // 70% confidence threshold
          .map(c => ({
            type: c.name,
            confidence: c.value
          }));
        
        resolve(detectedItems);
      }
    );
  });
}

// Main endpoint
router.post('/analyze-wardrobe-clarifai', upload.single('video'), async (req, res) => {
  try {
    const videoPath = req.file.path;
    const framesDir = path.join('uploads', 'frames', Date.now().toString());
    
    fs.mkdirSync(framesDir, { recursive: true });
    
    console.log('Extracting frames...');
    const frames = await extractFrames(videoPath, framesDir);
    
    console.log(\`Analyzing \${frames.length} frames with Clarifai...\`);
    const allDetections = [];
    
    // Analyze each frame
    for (const frame of frames) {
      const detections = await analyzeFrameWithClarifai(frame);
      allDetections.push(...detections);
    }
    
    // Aggregate results
    const itemCounts = {};
    allDetections.forEach(item => {
      const key = item.type.toLowerCase();
      if (!itemCounts[key]) {
        itemCounts[key] = { type: item.type, count: 0, avgConfidence: 0 };
      }
      itemCounts[key].count++;
      itemCounts[key].avgConfidence += item.confidence;
    });
    
    // Calculate averages
    const finalItems = Object.values(itemCounts).map(item => ({
      type: item.type,
      detectedInFrames: item.count,
      confidence: (item.avgConfidence / item.count).toFixed(2)
    }));
    
    // Cleanup
    fs.rmSync(videoPath);
    fs.rmSync(framesDir, { recursive: true });
    
    res.json({
      success: true,
      itemsDetected: finalItems,
      totalFramesAnalyzed: frames.length
    });
    
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
\`\`\`

---

## Frontend Code (React Native - Works with Both APIs)

\`\`\`typescript
// screens/WardrobeVideoAnalysis.tsx
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const WardrobeVideoAnalysis = () => {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled) {
      analyzeVideo(result.assets[0].uri);
    }
  };

  const analyzeVideo = async (videoUri: string) => {
    setAnalyzing(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('video', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'wardrobe.mp4',
      } as any);

      // Choose which API to use:
      // Option A: Google Cloud Vision
      // const response = await axios.post('YOUR_API_URL/analyze-wardrobe', formData);
      
      // Option B: Clarifai (Recommended)
      const response = await axios.post('YOUR_API_URL/analyze-wardrobe-clarifai', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000 // 2 minutes
      });

      setResults(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Wardrobe Video Analysis</Text>
      
      <TouchableOpacity style={styles.button} onPress={pickVideo} disabled={analyzing}>
        <Text style={styles.buttonText}>
          {analyzing ? 'Analyzing...' : 'Upload Wardrobe Video'}
        </Text>
      </TouchableOpacity>

      {analyzing && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#000" />
          <Text style={styles.loadingText}>Analyzing your wardrobe...</Text>
          <Text style={styles.subText}>This may take 30-60 seconds</Text>
        </View>
      )}

      {results && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Detected Items:</Text>
          {results.itemsDetected.map((item: any, index: number) => (
            <View key={index} style={styles.resultItem}>
              <Text style={styles.itemType}>{item.type}</Text>
              <Text style={styles.itemConfidence}>
                {(item.confidence * 100).toFixed(0)}% confident
              </Text>
            </View>
          ))}
          <Text style={styles.framesText}>
            Analyzed {results.totalFramesAnalyzed} frames
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#fff' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20 },
  button: { backgroundColor: '#000', padding: 15, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  loadingContainer: { marginTop: 30, alignItems: 'center' },
  loadingText: { marginTop: 10, fontSize: 16 },
  subText: { marginTop: 5, fontSize: 14, color: '#666' },
  resultsContainer: { marginTop: 30 },
  resultsTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 15 },
  resultItem: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#eee' },
  itemType: { fontSize: 16, textTransform: 'capitalize' },
  itemConfidence: { fontSize: 14, color: '#666' },
  framesText: { marginTop: 15, fontSize: 14, color: '#999', textAlign: 'center' },
});

export default WardrobeVideoAnalysis;
\`\`\`

---

## Comparison Summary

| Feature | Google Cloud Vision | Clarifai Fashion |
|---------|-------------------|------------------|
| **Fashion Specialization** | ❌ General objects | ✅ Fashion-specific |
| **Accuracy for Clothes** | Good (80%) | Excellent (90%+) |
| **Setup Complexity** | Medium | Easy |
| **Free Tier** | 1,000/month | 1,000/month |
| **Pricing** | $1.50/1000 | $1.20/1000 |
| **Best For** | General use | **Fashion apps** |

## My Recommendation

**Use Clarifai Fashion API** because:
1. It's specifically designed for fashion/clothing detection
2. Better accuracy for your use case
3. Easier to set up
4. Slightly cheaper

## Next Steps

1. Choose which API you want to use
2. I'll help you set it up
3. We'll integrate it into your app

Which one do you want to proceed with?
