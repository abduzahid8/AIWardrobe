/**
 * setup-tryon-bucket.js
 *
 * Creates the public `try-on-snapshots` Supabase Storage bucket
 * required by the mannequin 3D try-on flow.
 *
 * Usage:
 *   SUPABASE_SERVICE_ROLE_KEY=<your_key> node scripts/setup-tryon-bucket.js
 *
 * Get your service role key from:
 *   Supabase Dashboard → Settings → API → service_role (secret)
 */

require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SERVICE_ROLE_KEY) {
    console.error('❌ Missing EXPO_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env');
    console.error('   Run: SUPABASE_SERVICE_ROLE_KEY=<key> node scripts/setup-tryon-bucket.js');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

async function main() {
    console.log('🔧 Setting up try-on-snapshots bucket...');

    const { data: existing, error: listErr } = await supabase.storage.listBuckets();
    if (listErr) {
        console.error('❌ Failed to list buckets:', listErr.message);
        process.exit(1);
    }

    const bucketExists = existing.some(b => b.name === 'try-on-snapshots');

    if (bucketExists) {
        console.log('✅ Bucket "try-on-snapshots" already exists');
    } else {
        const { error: createErr } = await supabase.storage.createBucket('try-on-snapshots', {
            public: true,
            allowedMimeTypes: ['image/jpeg', 'image/png', 'image/webp'],
            fileSizeLimit: 5 * 1024 * 1024, // 5 MB max
        });

        if (createErr) {
            console.error('❌ Failed to create bucket:', createErr.message);
            process.exit(1);
        }

        console.log('✅ Bucket "try-on-snapshots" created successfully (public)');
    }

    console.log('\n📋 Next: Add this RLS policy in Supabase Dashboard → Storage → try-on-snapshots → Policies:');
    console.log('   Allow authenticated users to INSERT into their own folder:');
    console.log('   CREATE POLICY "Users can upload their own snapshots"');
    console.log('   ON storage.objects FOR INSERT TO authenticated');
    console.log('   WITH CHECK (bucket_id = \'try-on-snapshots\' AND auth.uid()::text = (storage.foldername(name))[1]);');
}

main().catch(console.error);
