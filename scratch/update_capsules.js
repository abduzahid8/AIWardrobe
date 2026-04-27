const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.EXPO_PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY
);

async function fix() {
  const { data, error } = await supabase.from('featured_capsules')
      .update({ image_url: "https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop-catalog/admin/inspo-66da23a0-ab32-43e7-8c61-7208a0d6caf2-1777201066061.png" })
      .eq('id', '66da23a0-ab32-43e7-8c61-7208a0d6caf2');
  console.log("Update result:", data, error);
}

fix();
