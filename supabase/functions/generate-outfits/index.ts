import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
    // Handle CORS preflight request
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    try {
        const { occasion, stylePreferences, wardrobeItems, weather, limit = 3 } = await req.json()

        if (!occasion && !stylePreferences) {
            return new Response(
                JSON.stringify({ error: "Please provide an occasion or style preferences" }),
                { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
            )
        }

        const openAiApiKey = Deno.env.get('OPENAI_API_KEY')
        if (!openAiApiKey) {
            throw new Error("OPENAI_API_KEY is not configured in the edge function environment.");
        }

        console.log(`Generating AI outfits for Occasion: ${occasion}, Style: ${stylePreferences}, Items Available: ${wardrobeItems?.length || 0}`);

        // System Prompt Design (The "Brain" of the Alta Daily-like AI Stylist)
        const systemPrompt = `You are a world-class personal AI fashion stylist, comparable to the best celebrity stylists. 
Your goal is to build stylish, cohesive, and occasion-appropriate outfits exclusively using the items provided in the user's digital wardrobe.

CRITICAL RULES:
1. You MUST ONLY recommend items that are present in the provided \`wardrobeItems\` list. Do not invent items they don't own, unless specifically noting a "missing piece" in the styling tips.
2. Consider the Occasion, Style Preferences, and Weather (if provided) carefully.
3. Ensure color harmony, balanced silhouettes, and appropriate formality.
4. Return exactly ${limit} outfit recommendation(s).
5. You must respond in pure JSON format matching the schema below. Do not include markdown formatting like \`\`\`json.

EXPECTED JSON OUTPUT SCHEMA:
{
  "outfits": [
    {
      "id": "unique-string-uuid",
      "occasion": "The event/occasion this is for",
      "style": "The overarching style (e.g., Casual Chic)",
      "description": "A vivid 1-2 sentence description of the vibe",
      "confidence": 0.95, // Your confidence score 0.0 to 1.0
      "items": [
        {
          "id": "THE_EXACT_ID_FROM_WARDROBE_ITEMS",
          "type": "e.g., top, bottom, shoes",
          "color": "e.g., black",
          "recommendation": "Why this piece works here"
        }
      ],
      "stylingTips": [
        "A useful styling trick (e.g., French tuck the shirt)",
        "An accessory suggestion"
      ]
    }
  ]
}`;

        // Construct User Prompt Context
        const userPrompt = `
Context:
Occasion/Event: ${occasion || 'Everyday wear'}
Style Preferences: ${stylePreferences || 'None specified'}
Weather: ${weather ? `${weather.temp}°C, ${weather.condition}` : 'Unknown/Indoor'}
Items in Closet: ${JSON.stringify(wardrobeItems || [], null, 2)}

Generate ${limit} distinct outfits using ONLY the closet items above. If the closet is empty or lacks sufficient items, generate an ideal outfit and note in the styling tips that they are missing key pieces.`

        // Call OpenAI API
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${openAiApiKey}`,
            },
            body: JSON.stringify({
                model: 'gpt-4o-mini', // Fast and cost-effective for JSON structuring
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ],
                temperature: 0.7,
                response_format: { type: "json_object" }
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("OpenAI API Error:", errorText);
            throw new Error(`OpenAI API returned status ${response.status}`);
        }

        const data = await response.json();
        const aiResponseContent = data.choices[0].message.content;

        let parsedOutfits;
        try {
            parsedOutfits = JSON.parse(aiResponseContent);
            if (!parsedOutfits.outfits || !Array.isArray(parsedOutfits.outfits)) {
                throw new Error("Invalid output structure from AI.");
            }
        } catch (parseError) {
            console.error("Failed to parse AI response:", aiResponseContent);
            throw new Error("AI returned malformed JSON.");
        }

        return new Response(
            JSON.stringify({
                success: true,
                outfits: parsedOutfits.outfits,
                matchCount: parsedOutfits.outfits.length
            }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
        )

    } catch (error: any) {
        console.error("Generate Outfits Edge Function Error:", error);
        return new Response(
            JSON.stringify({ error: error.message }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
        )
    }
})
