import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    // Authenticate the requesting user
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(JSON.stringify({ error: 'Missing authorization header' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;

    // User client — to identify who is making the request
    const userClient = createClient(supabaseUrl, Deno.env.get('SUPABASE_ANON_KEY')!, {
      global: { headers: { Authorization: authHeader } },
    });

    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) {
      return new Response(JSON.stringify({ error: 'Invalid or expired token' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const userId = user.id;

    // Admin client — service role to delete user data and auth record
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // Delete user data from all tables — child tables first due to foreign key constraints
    const tableDeletes: Array<{ table: string; col: string }> = [
      { table: 'payments', col: 'user_id' },
      { table: 'subscriptions', col: 'user_id' },
      { table: 'wear_logs', col: 'user_id' },
      { table: 'saved_outfits', col: 'user_id' },
      { table: 'clothing_items', col: 'user_id' },
      { table: 'profiles', col: 'id' },
    ];

    for (const { table, col } of tableDeletes) {
      const { error } = await adminClient.from(table).delete().eq(col, userId);
      if (error) {
        console.error(`Failed to delete from ${table}:`, error);
        return new Response(JSON.stringify({ error: `Failed to delete data from ${table}` }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }
    }

    // Delete user uploads from storage (user_uploads and avatars buckets)
    const bucketsToClean = [
      { bucket: 'user_uploads', prefix: userId },
      { bucket: 'avatars', prefix: userId },
    ];

    for (const { bucket, prefix } of bucketsToClean) {
      const { data: files } = await adminClient.storage.from(bucket).list(prefix);
      if (files && files.length > 0) {
        const filePaths = files.map((f: { name: string }) => `${prefix}/${f.name}`);
        await adminClient.storage.from(bucket).remove(filePaths);
      }
    }

    // Delete the auth user record
    const { error: deleteError } = await adminClient.auth.admin.deleteUser(userId);
    if (deleteError) {
      console.error('Failed to delete auth user:', deleteError);
      return new Response(JSON.stringify({ error: 'Failed to delete account' }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Account deletion error:', error);
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});
