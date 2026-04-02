#!/bin/bash
# API Health Check Script
# Run this to test if the Edge Function is accessible

echo "=== AI Wardrobe API Health Check ==="
echo ""

# Get these values from your .env file
SUPABASE_URL="${EXPO_PUBLIC_SUPABASE_URL:-}"
SUPABASE_ANON_KEY="${EXPO_PUBLIC_SUPABASE_ANON_KEY:-}"

if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_ANON_KEY" ]; then
    echo "❌ Missing environment variables"
    echo "Please ensure EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY are set"
    exit 1
fi

echo "Supabase URL: ${SUPABASE_URL:0:40}..."
echo ""

# Test 1: Check Supabase REST API (auth)
echo "1️⃣ Testing Supabase REST API..."
curl -s -o /dev/null -w "%{http_code}" \
    "${SUPABASE_URL}/rest/v1/" \
    -H "apikey: ${SUPABASE_ANON_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_ANON_KEY}"

echo ""
# Test 2: Testing ai-process Edge Function with valid base64 image
echo "2️⃣ Testing ai-process Edge Function..."
curl -s -X POST \
    "${SUPABASE_URL}/functions/v1/ai-process" \
    -H "Authorization: Bearer ${SUPABASE_ANON_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==","operation":"classify"}' \
    | head -300

echo ""
echo "=== End of Health Check ==="
