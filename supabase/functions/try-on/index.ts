
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Replicate configuration
const REPLICATE_API_TOKEN = Deno.env.get('REPLICATE_API_TOKEN')

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

        // Verify User
        const {
            data: { user },
        } = await supabaseClient.auth.getUser()

        if (!user) {
            throw new Error("User not authenticated")
        }

        const { person_image, garment_image, garment_type } = await req.json()

        if (!person_image || !garment_image) {
            throw new Error("Missing person_image or garment_image")
        }

        // Call Replicate if token exists
        if (REPLICATE_API_TOKEN) {
            console.log("Calling Replicate for Try-On")

            // IDM-VTON model
            // https://replicate.com/cuuupid/idm-vton
            const model = "cuuupid/idm-vton:c871bb9b046607b6804fe43f38006d649989acf3333333333333333333333333" // Check exact version if needed, or use latest

            // Note: Replicate SDK is not in Deno standard lib easily, use fetch
            const response = await fetch("https://api.replicate.com/v1/predictions", {
                method: "POST",
                headers: {
                    "Authorization": `Token ${REPLICATE_API_TOKEN}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    version: "c871bb9b046607b6804fe43f38006d649989acf3333333333333333333333333", // Example hash
                    input: {
                        human_img: person_image,
                        garm_img: garment_image,
                        garment_des: "clothing", // Optional
                        category: garment_type === 'lower_body' ? 'lower_body' : 'upper_body',
                        n_samples: 1,
                        seed: 42
                    }
                })
            });

            if (!response.ok) {
                const err = await response.text();
                console.error("Replicate Error:", err);
                // Fallback to mock if API fails? Or throw.
                throw new Error("AI Try-On Service Error: " + err);
            }

            const prediction = await response.json();

            // Poll for result
            let result = prediction;
            while (result.status !== "succeeded" && result.status !== "failed") {
                await new Promise(r => setTimeout(r, 2000));
                const pollRes = await fetch(result.urls.get, {
                    headers: { "Authorization": `Token ${REPLICATE_API_TOKEN}` }
                });
                result = await pollRes.json();
            }

            if (result.status === "failed") {
                throw new Error("AI Processing Failed");
            }

            const outputImage = result.output; // Usually a URL string or array

            return new Response(
                JSON.stringify({
                    success: true,
                    resultImage: Array.isArray(outputImage) ? outputImage[0] : outputImage,
                    methodUsed: 'replicate'
                }),
                { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
            )

        } else {
            // MOCK Response
            console.log("No Replicate Token, using Mock");
            await new Promise(r => setTimeout(r, 2000)); // Simulate delay

            // Return the input garment image overlaying the person image? 
            // Or just return the garment image as a placeholder?
            // Let's just return the garment image for now to show "something" happened, 
            // or a placeholder URL if we had one.

            return new Response(
                JSON.stringify({
                    success: true,
                    resultImage: garment_image, // Just returning garment as mock result
                    methodUsed: 'mock',
                    note: "Add REPLICATE_API_TOKEN to Supabase Secrets for real AI."
                }),
                { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
            )
        }

    } catch (error: any) {
        return new Response(
            JSON.stringify({ error: error.message }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
        )
    }
})
