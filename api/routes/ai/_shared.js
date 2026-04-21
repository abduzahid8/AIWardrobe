/**
 * Shared imports and service initializations for AI routes
 * Now uses HuggingFace as the single AI provider (free tier).
 */
import express from "express";
import axios from "axios";
import { HfInference } from "@huggingface/inference";
import { createClient } from "@supabase/supabase-js";
import cosineSimilarity from "compute-cosine-similarity";

import { authenticateToken } from "../../middleware/auth.js";
import { requireTier } from "../../middleware/subscriptionGuard.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import { ALICEVISION_URL } from "../../config.js";
import logger from "../../utils/logger.js";
import { validateAIChat, validateImageData } from "../../middleware/validators.js";
import hfService from "../../services/huggingface.js";

// Service singletons
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const hf = new HfInference(process.env.HF_TOKEN);

export {
  express, axios, HfInference,
  createClient, cosineSimilarity,
  authenticateToken, requireTier, aiLimiter, ALICEVISION_URL, logger,
  validateAIChat, validateImageData,
  supabase, hf, hfService,
};
