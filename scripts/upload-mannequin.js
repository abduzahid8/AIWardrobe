#!/usr/bin/env node
/**
 * upload-mannequin.js
 *
 * Uploads mannequin_male.glb (or any .glb you specify) to Supabase Storage
 * and prints the public URL to use in mannequinConfig.ts
 *
 * Usage:
 *   node scripts/upload-mannequin.js [path-to-file.glb]
 *
 * Requirements:
 *   npm install @supabase/supabase-js
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

// ── Config ──────────────────────────────────────────────────────────────────
const SUPABASE_URL      = process.env.EXPO_PUBLIC_SUPABASE_URL
                        || 'https://fyqpifmrsftsfqibhwhy.supabase.co';
// Use service role key for upload (NOT anon key). Set as env var for security.
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || '';
const BUCKET_NAME       = 'models';
const REMOTE_FILE_NAME  = 'mannequin_male.glb';

// ── File path ────────────────────────────────────────────────────────────────
const localFile = process.argv[2]
  || path.join(__dirname, '..', 'assets', 'models', 'mannequin_male.glb');

// ────────────────────────────────────────────────────────────────────────────
async function main() {
  if (!SUPABASE_SERVICE_KEY) {
    console.error('\n❌  SUPABASE_SERVICE_KEY env var is not set.');
    console.error('   Get it from: Supabase Dashboard → Project Settings → API → service_role key');
    console.error('   Then run:  SUPABASE_SERVICE_KEY=your_key node scripts/upload-mannequin.js\n');
    process.exit(1);
  }

  if (!fs.existsSync(localFile)) {
    console.error('\n❌  File not found:', localFile);
    console.error('   Export your .blend → .glb from Blender first, then place it at:', localFile, '\n');
    process.exit(1);
  }

  const fileSizeMB = (fs.statSync(localFile).size / 1024 / 1024).toFixed(1);
  console.log(`\n📦 Uploading: ${path.basename(localFile)} (${fileSizeMB} MB)`);
  console.log(`   Bucket:     ${SUPABASE_URL}/storage/v1/${BUCKET_NAME}/${REMOTE_FILE_NAME}`);

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

  // Create bucket if it doesn't exist (public)
  const { error: bucketError } = await supabase.storage.createBucket(BUCKET_NAME, { public: true });
  if (bucketError && !bucketError.message.includes('already exists')) {
    console.error('❌  Bucket error:', bucketError.message);
    process.exit(1);
  }

  const fileBuffer = fs.readFileSync(localFile);

  const { error: uploadError } = await supabase.storage
    .from(BUCKET_NAME)
    .upload(REMOTE_FILE_NAME, fileBuffer, {
      contentType: 'model/gltf-binary',
      upsert: true,
    });

  if (uploadError) {
    console.error('❌  Upload failed:', uploadError.message);
    process.exit(1);
  }

  const { data: { publicUrl } } = supabase.storage
    .from(BUCKET_NAME)
    .getPublicUrl(REMOTE_FILE_NAME);

  console.log('\n✅  Upload successful!');
  console.log('\n📋  Public URL (paste this into mannequinConfig.ts):');
  console.log('\n   ' + publicUrl + '\n');
}

main().catch(console.error);
