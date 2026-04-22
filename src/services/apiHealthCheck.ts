/**
 * API Health Check - Quick diagnostic for testing API connectivity
 */

import { supabase } from '../lib/supabase';
import { createLogger } from '../utils/logger';

const logger = createLogger('ApiHealth');

export interface ApiHealthResult {
  supabaseAuth: boolean;
  edgeFunction: boolean;
  edgeFunctionError?: string;
  timestamp: string;
}

/**
 * Quick health check - tests Supabase auth and Edge Function availability
 */
export async function checkApiHealth(): Promise<ApiHealthResult> {
  const result: ApiHealthResult = {
    supabaseAuth: false,
    edgeFunction: false,
    timestamp: new Date().toISOString(),
  };

  // Test 1: Check Supabase auth session
  try {
    const { data: sessionData } = await supabase.auth.getSession();
    result.supabaseAuth = !!sessionData.session;
    logger.info(`Supabase auth: ${result.supabaseAuth ? '✓' : '✗ (no session)'}`);
  } catch (err) {
    console.error('[API Health] Supabase auth error:', err);
  }

  // Test 2: Try to invoke the ai-process Edge Function with minimal payload
  try {
    const { data, error } = await supabase.functions.invoke('ai-process', {
      body: { 
        image: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD', // 1x1 pixel test image
        operation: 'classify'
      },
    });

    if (error) {
      result.edgeFunctionError = error.message;
      console.error('[API Health] Edge Function error:', error.message);
    } else {
      result.edgeFunction = data?.success || false;
      logger.info(`Edge Function: ${result.edgeFunction ? '✓' : '✗ (returned error)'}`);
      if (data?.error) {
        result.edgeFunctionError = data.error;
        logger.warn('Edge Function response error', data.error);
      }
    }
  } catch (err) {
    result.edgeFunctionError = err instanceof Error ? err.message : 'Unknown error';
    console.error('[API Health] Edge Function exception:', err);
  }

  // Summary
  logger.info('Result', result);
  return result;
}

/**
 * Log all API configuration for debugging
 */
export function logApiConfig(): void {
  logger.info('Supabase URL', process.env.EXPO_PUBLIC_SUPABASE_URL?.slice(0, 30) + '...');
  logger.info('Anon Key present', !!process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY);
}
