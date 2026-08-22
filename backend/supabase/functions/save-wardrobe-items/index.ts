
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { decode } from "https://deno.land/std@0.168.0/encoding/base64.ts"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    try {
        const supabaseClient = createClient(
            Deno.env.get('SUPABASE_URL') ?? '',
            Deno.env.get('SUPABASE_ANON_KEY') ?? '',
            { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
        )

        const {
            data: { user },
        } = await supabaseClient.auth.getUser()

        if (!user) {
            throw new Error("User not authenticated")
        }

        const { items } = await req.json()

        if (!items || !Array.isArray(items)) {
            throw new Error("Invalid items array")
        }

        const savedItems = []

        for (const item of items) {
            // Upload image if base64
            let imageUrl = item.image;
            if (item.image && item.image.startsWith('data:image')) {
                try {
                    const base64Data = item.image.split(',')[1];
                    const fileExt = 'jpg'; // Assume jpg for simplicity or extract from data uri
                    const filePath = `${user.id}/${Date.now()}_${Math.random().toString(36).substring(7)}.${fileExt}`;

                    const imageBuffer = decode(base64Data);

                    const { error: uploadError } = await supabaseClient
                        .storage
                        .from('wardrobe')
                        .upload(filePath, imageBuffer, {
                            contentType: 'image/jpeg',
                            upsert: false
                        });

                    if (uploadError) {
                        console.error("Upload Error:", uploadError);
                        // If bucket doesn't exist, maybe try creating it? Or fail.
                        // Assuming 'wardrobe' bucket exists.
                        throw uploadError;
                    }

                    const { data: { publicUrl } } = supabaseClient
                        .storage
                        .from('wardrobe')
                        .getPublicUrl(filePath);

                    imageUrl = publicUrl;

                } catch (e) {
                    console.error("Failed to upload image", e);
                    // Skip image or fail? Continue with null image?
                    imageUrl = null;
                }
            }

            savedItems.push({
                user_id: user.id,
                type: item.type || 'Unknown',
                category: item.category || item.type || 'Unknown',
                color: item.color || 'Unknown',
                style: item.style || 'Casual',
                description: item.description,
                material: item.material,
                season: 'All Seasons',
                image_url: imageUrl,
                created_at: new Date().toISOString()
            })
        }

        if (savedItems.length > 0) {
            const { error } = await supabaseClient
                .from('clothing_items')
                .insert(savedItems);

            if (error) throw error;
        }

        return new Response(
            JSON.stringify({ success: true, count: savedItems.length }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
        )

    } catch (error: any) {
        return new Response(
            JSON.stringify({ error: error.message }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
        )
    }
})
