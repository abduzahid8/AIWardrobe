/**
 * AIWardrobe API Server
 * 
 * Main entry point for the Express server.
 * All routes are modularized and imported from ./routes
 */

import express from "express";
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
  wardrobeRouter, chatRouter, outfitsRouter,
  imageProcessorRouter, studioRouter
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
import tryonRenderRoutes from "./routes/tryon.js";
import tryonGeminiRoutes from "./routes/tryon-gemini.js";
// Legacy try-on routes (tryon-v1/v2/v3) are intentionally NOT mounted.
// Try-on is admin-only; only the canonical /api/tryon/render is exposed.

// Import middleware
import { apiLimiter, aiLimiter } from "./middleware/rateLimit.js";
import { auditLogger } from "./middleware/security.js";
import { authenticateToken, requireAdmin } from "./middleware/auth.js";

import logger from "./utils/logger.js";
import "./lib/supabase.js"; // Initialize Supabase client globally

const app = express();
const port = process.env.PORT || 3000;

// Trust the first proxy hop (Render / Cloudflare) so req.ip and X-Forwarded-For
// resolve to the real client IP. Required for rate limiting and audit logging.
app.set('trust proxy', 1);

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

// Body parsing with size limits (20MB max — try-on sends base64 mannequin + garments)
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ limit: "20mb", extended: true }));

// Apply general rate limiting to all routes
app.use(apiLimiter);

// Apply audit logging (for security-relevant actions)
app.use(auditLogger);

// ============================================
// SYSTEM SHUTDOWN
// ============================================

// Graceful shutdown handling
// Graceful shutdown is wired up below, after server.listen().

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
// (legacy /api/try-on route removed — all try-on now uses mannequin-locked FLUX.1-Kontext-dev at /api/tryon/render)
app.use("/api", aiLimiter, wardrobeRouter);       // /api/scan-wardrobe
app.use("/api", aiLimiter, chatRouter);           // /api/smart-search, /api/ai-chat
app.use("/api", aiLimiter, outfitsRouter);        // /api/generate-outfits
app.use("/api", aiLimiter, imageProcessorRouter); // /api/process-upload
app.use("/api/studio", aiLimiter, studioRouter);  // /api/studio/analyze, /api/studio/generate

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

// Gemini 2.0 Flash try-on — cheaper alternative to FLUX (~6-12x cheaper)
// Must be mounted BEFORE generic /api/tryon to avoid being shadowed
// Quota is enforced client-side via useSubscriptionGate; any authenticated user may call this.
app.use("/api/tryon/gemini", authenticateToken, aiLimiter, tryonGeminiRoutes);

// Deterministic mannequin try-on renderer (FLUX). Any authenticated user may call this.
app.use("/api/tryon", authenticateToken, aiLimiter, tryonRenderRoutes);

// No local DB seeding. Handled via Supabase directly.

// ============================================
// HEALTH CHECK
// ============================================

app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Readiness probe — distinct from liveness. Currently mirrors /health, but
// keeps a stable contract for load balancers / k8s-style probes.
let isShuttingDown = false;
app.get("/ready", (req, res) => {
  if (isShuttingDown) return res.status(503).json({ status: "shutting_down" });
  res.json({ status: "ready" });
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

const server = app.listen(port, "0.0.0.0", () => {
  logger.info(`🚀 Server running on port ${port}`);
  logger.info(`📍 Environment: ${process.env.NODE_ENV || 'development'}`);
});

// Defensive HTTP timeouts. Without these, hung upstream calls leak sockets.
// requestTimeout: total time a single request may take.
// headersTimeout:  must be > requestTimeout per Node docs.
// keepAliveTimeout: > LB idle timeout to avoid 502s from Render's proxy.
// NOTE: Mobile-VTON /tryon can take 30-60s for GPU inference, so 180s matches
// the mobile app's Axios timeout and gives headroom for cold-start + inference.
server.requestTimeout = 180_000;
server.headersTimeout = 185_000;
server.keepAliveTimeout = 65_000;

async function gracefulShutdown(signal) {
  if (isShuttingDown) return;
  isShuttingDown = true;
  logger.info(`🔄 ${signal} received — draining connections...`);
  const forceExit = setTimeout(() => {
    logger.error('Forced shutdown after 25s timeout');
    process.exit(1);
  }, 25_000).unref();
  server.close(err => {
    clearTimeout(forceExit);
    if (err) {
      logger.error('Error during shutdown:', err);
      process.exit(1);
    }
    logger.info('✅ Clean shutdown complete');
    process.exit(0);
  });
}
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('unhandledRejection', (reason) => {
  logger.error('unhandledRejection:', reason);
});
process.on('uncaughtException', (err) => {
  logger.error('uncaughtException:', err);
  gracefulShutdown('uncaughtException');
});
