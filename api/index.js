/**
 * AIWardrobe API Server
 * 
 * Main entry point for the Express server.
 * All routes are modularized and imported from ./routes
 */

import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import helmet from "helmet";
import "dotenv/config";

// Import route modules
import authRoutes from "./routes/auth.js";
import clothingRoutes from "./routes/clothing.js";
import outfitRoutes from "./routes/outfits.js";
import weatherRoutes from "./routes/weather.js";
import aiRoutes from "./routes/ai.js";
import statsRoutes from "./routes/stats.js";
import alicevisionRoutes from "./routes/alicevision.js";

// Import middleware
import { apiLimiter, aiLimiter } from "./middleware/rateLimit.js";

// Import models for seeding
import Outfit from "./models/outfit.js";
import { HfInference } from "@huggingface/inference";

const app = express();
const port = process.env.PORT || 3000;

// ============================================
// SECURITY MIDDLEWARE
// ============================================

// Security headers (Helmet)
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" }, // Allow images from other origins
  contentSecurityPolicy: false, // Disable CSP for API
}));

// CORS configuration
const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['*'];
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, Postman)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes('*') || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400 // Cache preflight for 24 hours
}));

// Body parsing with size limits
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Apply general rate limiting to all routes
app.use(apiLimiter);

// ============================================
// DATABASE CONNECTION
// ============================================

const MONGODB_URI = process.env.MONGODB_URI || process.env.MONGO_URI;

if (!MONGODB_URI) {
  console.error("❌ FATAL: MONGODB_URI environment variable is not set!");
  console.error("   Please add MONGODB_URI to your .env file");
  // Continue with fallback for backward compatibility
}

mongoose
  .connect(MONGODB_URI || "mongodb+srv://karimdzanovzoha:Abduzahid8@aiwardrobe.fah7ml3.mongodb.net/?appName=AIWardrobe")
  .then(() => console.log("✅ Connected to MongoDB"))
  .catch((err) => console.log("❌ Error connecting to MongoDB:", err));

// ============================================
// MOUNT ROUTES
// ============================================

// Authentication routes (has its own rate limiting)
app.use("/", authRoutes);

// Clothing item management
app.use("/clothing-items", clothingRoutes);
app.use("/wardrobe", clothingRoutes);

// Outfit management
app.use("/save-outfit", outfitRoutes);

// Weather API
app.use("/weather", weatherRoutes);

// Statistics & Analytics
app.use("/stats", statsRoutes);

// AI-powered features (with stricter rate limiting)
app.use("/api", aiLimiter, aiRoutes);
app.use("/", aiRoutes); // For routes like /try-on, /scan-wardrobe

// AliceVision computer vision microservice integration
app.use("/alicevision", aiLimiter, alicevisionRoutes);

// ============================================
// DATABASE SEEDING
// ============================================

const hf = new HfInference(process.env.HF_TOKEN);

const generateEmbedding = async (text) => {
  const response = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: text,
  });
  return response;
};

const seedData = async () => {
  try {
    const count = await Outfit.countDocuments();
    if (count === 0) {
      const outfits = [
        {
          occasion: "date",
          style: "casual",
          items: ["White linen shirt", "Dark jeans", "Loafers"],
          image: "https://i.pinimg.com/736x/b2/6e/c7/b26ec7bc30ca9459b918ae8f7bf66305.jpg",
        },
        {
          occasion: "date",
          style: "elegant",
          items: ["White flared pants", "sandals", "sunglasses"],
          image: "https://i.pinimg.com/736x/8c/61/12/8c6112457ae46fa1e0aea8b8f5ed18ec.jpg",
        },
        {
          occasion: "coffee",
          style: "casual",
          items: ["cropped t-shirt", "wide-leg beige trousers", "Samba sneakers"],
          image: "https://i.pinimg.com/736x/d7/2d/26/d72d268ca4ff150db1db560b25afb843.jpg",
        },
        {
          occasion: "interview",
          style: "formal",
          items: ["Light blue shirt", "wide-leg jeans", "Silver wristwatch"],
          image: "https://i.pinimg.com/736x/1c/50/bc/1c50bcef1b46efe5db4008252ea8cfa5.jpg",
        },
        {
          occasion: "beach",
          style: "beach",
          items: ["brown T shirt", "beige shorts", "Sunglasses"],
          image: "https://i.pinimg.com/1200x/86/57/59/8657592bd659335ffd081fdab10b87a4.jpg",
        },
      ];

      for (const outfit of outfits) {
        const text = `${outfit.occasion} ${outfit.style} ${outfit.items.join(", ")}`;
        const embedding = await generateEmbedding(text);
        await new Outfit({ ...outfit, embedding }).save();
      }
      console.log("✅ Database seeded with", outfits.length, "outfits");
    } else {
      console.log("✅ Database already has", count, "outfits");
    }
  } catch (err) {
    console.error("❌ Seeding failed:", err.message);
  }
};

seedData();

// ============================================
// HEALTH CHECK
// ============================================

app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ============================================
// START SERVER
// ============================================

app.listen(port, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${port}`);
  console.log(`📍 Routes mounted:`);
  console.log(`   - Auth: /register, /login, /me`);
  console.log(`   - Clothing: /clothing-items`);
  console.log(`   - Wardrobe: /wardrobe/add-batch`);
  console.log(`   - Outfits: /save-outfit`);
  console.log(`   - Weather: /weather`);
  console.log(`   - AI: /api/*, /try-on, /scan-wardrobe`);
  console.log(`   - AliceVision: /alicevision/keyframe, /alicevision/segment, /alicevision/lighting, /alicevision/process`);
});
