const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.EXPO_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY
);

async function test() {
  const { data, error } = await supabase.storage.from('shop-catalog').list('admin', {
    limit: 5,
    sortBy: { column: 'created_at', order: 'desc' }
  });
  console.log("Recent files:", JSON.stringify(data, null, 2));
  console.log("Error:", error);
}

test();
