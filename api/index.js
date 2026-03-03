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
import {
  analyzeRouter, clothingRouter, productPhotoRouter,
  tryonRouter, wardrobeRouter, chatRouter, outfitsRouter,
  imageProcessorRouter
} from "./routes/ai/index.js";
import statsRoutes from "./routes/stats.js";
import alicevisionRoutes from "./routes/alicevision.js";
import emailIngestionRoutes from "./routes/emailIngestion.js";
import tripPlannerRoutes from "./routes/tripPlanner.js";
import wardrobeAnalyticsRoutes from "./routes/wardrobeAnalytics.js";
import subscriptionRoutes from "./routes/subscription.js";
import geminiRoutes from "./routes/gemini.js";
import analyticsRoutes from "./routes/analytics.js";
import accountRoutes from "./routes/account.js";

// Import middleware
import { apiLimiter, aiLimiter } from "./middleware/rateLimit.js";
import { auditLogger } from "./middleware/security.js";

// Import models for seeding
import Outfit from "./models/outfit.js";
import { HfInference } from "@huggingface/inference";
import logger from "./utils/logger.js";

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

// CORS configuration — deny-by-default in production
const isProduction = process.env.NODE_ENV === 'production';
const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',').map(s => s.trim()) || [];

// In development, allow localhost origins for convenience
if (!isProduction && allowedOrigins.length === 0) {
  allowedOrigins.push('http://localhost:3000', 'http://localhost:8081', 'http://localhost:19006');
}

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, Postman, server-to-server)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else if (!isProduction) {
      // In dev, warn but allow
      console.warn(`CORS: allowing unlisted origin in dev: ${origin}`);
      callback(null, true);
    } else {
      callback(new Error(`Not allowed by CORS: ${origin}`));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400
}));

// Body parsing with size limits (10MB max to prevent memory exhaustion)
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ limit: "10mb", extended: true }));

// Apply general rate limiting to all routes
app.use(apiLimiter);

// Apply audit logging (for security-relevant actions)
app.use(auditLogger);

// ============================================
// DATABASE CONNECTION
// ============================================

const MONGODB_URI = process.env.MONGODB_URI || process.env.MONGO_URI;

if (!MONGODB_URI) {
  logger.error("❌ FATAL: MONGODB_URI environment variable is not set!");
  logger.error("   Please add MONGODB_URI to your .env file");
  logger.error("   Example: MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/dbname");
  process.exit(1);
}

mongoose
  .connect(MONGODB_URI, {
    maxPoolSize: 10,
    serverSelectionTimeoutMS: 5000,
    socketTimeoutMS: 45000,
  })
  .then(() => logger.info("✅ Connected to MongoDB"))
  .catch((err) => {
    logger.error("❌ Error connecting to MongoDB:", err.message);
    process.exit(1);
  });

// Graceful shutdown handling
process.on('SIGINT', async () => {
  logger.info('🔄 Gracefully shutting down...');
  await mongoose.connection.close();
  logger.info('✅ MongoDB connection closed');
  process.exit(0);
});

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

// AI-powered features (split into domain-specific routers)
app.use("/api", aiLimiter, analyzeRouter);      // /api/analyze-frames
app.use("/api", aiLimiter, clothingRouter);      // /api/openai/analyze-clothing, /api/openai/generate-image, etc.
app.use("/api", aiLimiter, productPhotoRouter);  // /api/product-photo/process, /api/v2/product-photo/process-multi
app.use("/api", aiLimiter, tryonRouter);          // /api/try-on
app.use("/api", aiLimiter, wardrobeRouter);       // /api/scan-wardrobe
app.use("/api", aiLimiter, chatRouter);           // /api/smart-search, /api/ai-chat
app.use("/api", aiLimiter, outfitsRouter);        // /api/generate-outfits
app.use("/api", aiLimiter, imageProcessorRouter); // /api/process-upload

// Gemini AI proxy (server-side key — never exposed to client)
app.use("/api/gemini", aiLimiter, geminiRoutes);

// AliceVision computer vision microservice integration
app.use("/alicevision", aiLimiter, alicevisionRoutes);

// Email receipt ingestion (competitive advantage feature)
app.use("/api/email", emailIngestionRoutes);

// Trip planner (engagement feature)
app.use("/api/trip-planner", tripPlannerRoutes);

// Wardrobe analytics (cost-per-wear tracking)
app.use("/api/wardrobe-analytics", wardrobeAnalyticsRoutes);

// Subscription & Payment management
app.use("/api/subscription", subscriptionRoutes);

// Analytics event ingestion
app.use("/api/analytics", analyticsRoutes);

// Account management (deletion, GDPR)
app.use("/api/account", accountRoutes);

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
      logger.info("✅ Database seeded with", outfits.length, "outfits");
    } else {
      logger.info("✅ Database already has", count, "outfits");
    }
  } catch (err) {
    logger.error("❌ Seeding failed:", err.message);
  }
};

// Only seed when explicitly requested (e.g. SEED_DB=true npm start)
// Never runs automatically in production to avoid cold-start latency
if (process.env.SEED_DB === 'true') {
  seedData();
}

// ============================================
// HEALTH CHECK
// ============================================

app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ============================================
// GLOBAL ERROR HANDLER
// Prevents error.message/stack leaking to clients in production
// ============================================

app.use((err, req, res, _next) => {
  logger.error(`Unhandled error on ${req.method} ${req.path}:`, err.message);

  const isProduction = process.env.NODE_ENV === 'production';

  res.status(err.status || 500).json({
    error: isProduction ? 'Internal server error' : err.message,
    ...(isProduction ? {} : { stack: err.stack }),
  });
});

// ============================================
// START SERVER
// ============================================

app.listen(port, "0.0.0.0", () => {
  logger.info(`🚀 Server running on port ${port}`);
  logger.info(`📍 Environment: ${process.env.NODE_ENV || 'development'}`);
});
