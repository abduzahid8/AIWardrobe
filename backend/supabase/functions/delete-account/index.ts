import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.117.3';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};
const json = (body: Record<string, unknown>, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
});

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });
  if (req.method !== 'POST') return json({ error: 'Method not allowed' }, 405);

  try {
    const authorization = req.headers.get('Authorization');
    if (!authorization) return json({ error: 'Missing authorization header' }, 401);
    const url = Deno.env.get('SUPABASE_URL')!;
    const userClient = createClient(url, Deno.env.get('SUPABASE_ANON_KEY')!, {
      global: { headers: { Authorization: authorization } },
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) return json({ error: 'Invalid or expired token' }, 401);

    const admin = createClient(url, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, {
      auth: { persistSession: false, autoRefreshToken: false },
    });

    // Enumerate authoritative ownership across all buckets and nested paths.
    // Delete each page before reading the next; offsets would skip removed rows.
    // Preserve relational data until file cleanup has actually succeeded.
    for (let page = 0; page < 40; page++) {
      const { data: files, error } = await admin.rpc('list_account_deletion_objects', { p_user_id: user.id });
      if (error || !Array.isArray(files)) {
        console.error('[delete-account] Storage enumeration failed', { code: error?.code });
        return json({ error: 'Unable to clean up account files. Please try again.' }, 503);
      }
      if (files.length === 0) break;
      const buckets = new Map<string, string[]>();
      for (const file of files as Array<{ bucket_id: string; name: string }>) {
        const paths = buckets.get(file.bucket_id) || [];
        paths.push(file.name);
        buckets.set(file.bucket_id, paths);
      }
      for (const [bucket, paths] of buckets) {
        const { error: removeError } = await admin.storage.from(bucket).remove(paths);
        if (removeError) {
          console.error('[delete-account] Storage removal failed', { status: removeError.status });
          return json({ error: 'Unable to remove account files. Please try again.' }, 503);
        }
      }
    }

    // Never claim success with files left after bounded work/concurrent uploads.
    const { data: remaining, error: remainingError } = await admin.rpc('list_account_deletion_objects', { p_user_id: user.id });
    if (remainingError || !Array.isArray(remaining) || remaining.length > 0) {
      return json({ error: 'Account file cleanup is incomplete. Please try again.' }, 503);
    }

    // Verified foreign keys cascade profile, wardrobe, saved outfits, wear logs,
    // subscriptions, payments, body profiles and referrals in one transaction.
    const { error: deleteError } = await admin.auth.admin.deleteUser(user.id);
    if (deleteError) {
      console.error('[delete-account] Auth deletion failed', { code: deleteError.code });
      return json({ error: 'Unable to delete account. Please try again.' }, 500);
    }
    return json({ success: true });
  } catch {
    console.error('[delete-account] Unexpected failure');
    return json({ error: 'Unable to delete account. Please try again.' }, 500);
  }
});
