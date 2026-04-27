const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || 'YOUR_SUPABASE_URL';
const supabaseKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || 'YOUR_SUPABASE_KEY';

// Load from .env if possible
require('dotenv').config();

async function run() {
  const supabase = createClient(process.env.EXPO_PUBLIC_SUPABASE_URL, process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY);
  const { data, error } = await supabase.storage.from('shop-catalog').list('admin', { limit: 10 });
  console.log("FILES IN admin/:", data);
}
run();
