import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import logger from '../utils/logger.js';

dotenv.config();

const supabaseUrl = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
    logger.error("❌ FATAL: Supabase URL or Key environment variables are missing!");
    process.exit(1);
}

// Create a single supabase client for the entire API server
export const supabase = createClient(supabaseUrl, supabaseKey);

logger.info("✅ Supabase PG client initialized");

export default supabase;
