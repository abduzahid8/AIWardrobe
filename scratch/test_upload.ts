import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

dotenv.config();
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);

async function testUpload() {
  const filePath = `admin/test-node-${Date.now()}.txt`;
  console.log('Uploading to:', filePath);
  const blob = new Blob(["test info"], { type: "text/plain" });

  const { data, error } = await supabase.storage.from('shop-catalog').upload(filePath, blob, {
    contentType: 'text/plain'
  });

  if (error) {
    console.error('Upload failed:', error);
  } else {
    console.log('Upload success:', data);
  }
}

testUpload();
