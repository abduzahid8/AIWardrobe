/**
 * Shared imports and service initializations for AI routes
 */
import express from "express";
import axios from "axios";
import Replicate from "replicate";
import { HfInference } from "@huggingface/inference";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from "@supabase/supabase-js";
import cosineSimilarity from "compute-cosine-similarity";
import Outfit from "../../models/outfit.js";
import { authenticateToken } from "../../middleware/auth.js";
import { requireTier } from "../../middleware/subscriptionGuard.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import { ALICEVISION_URL } from "../../config.js";
import logger from "../../utils/logger.js";
import { validateAIChat, validateImageData } from "../../middleware/validators.js";

// Service singletons
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const hf = new HfInference(process.env.HF_TOKEN);
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

export {
  express, axios, Replicate, HfInference, GoogleGenerativeAI,
  createClient, cosineSimilarity, Outfit,
  authenticateToken, requireTier, aiLimiter, ALICEVISION_URL, logger,
  validateAIChat, validateImageData,
  supabase, genAI, hf, replicate,
};
