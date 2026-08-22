import express from "express";
import fetch from "node-fetch";

const router = express.Router();

/**
 * POST /api/studio/analyze
 * Analyzes a garment image using Gemini 2.5 Pro and generates a Flux prompt.
 */
router.post("/analyze", async (req, res) => {
  const { image, mediaType } = req.body;
  const GEMINI_KEY = process.env.GEMINI_API_KEY;

  if (!GEMINI_KEY) return res.status(500).json({ error: "GEMINI_API_KEY not set" });
  if (!image) return res.status(400).json({ error: "No image provided" });

  try {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GEMINI_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        generationConfig: {
          responseMimeType: "application/json",
          temperature: 0.2,
          responseSchema: {
            type: "object",
            properties: {
              garmentType: { type: "string" },
              color: { type: "string" },
              material: { type: "string" },
              details: { type: "string" },
              category: { type: "string" },
              generationPrompt: { type: "string" }
            },
            required: ["garmentType", "color", "material", "details", "category", "generationPrompt"]
          }
        },
        contents: [{
          role: "user",
          parts: [
            { inlineData: { mimeType: mediaType || "image/jpeg", data: image } },
            {
              text: `You are an expert fashion stylist and luxury e-commerce product photographer for brands like Massimo Dutti.
Analyze this clothing photo carefully. Focus on the main garment only.

Respond ONLY with a JSON object containing these specific fields:
{
  "garmentType": "specific garment name",
  "color": "precise color with hex if possible",
  "material": "fabric/material description",
  "details": "key design details: collar, buttons, pockets, zippers, stitching, fit",
  "category": "top / bottom / dress / outerwear / accessory / shoes",
  "generationPrompt": "Write the most detailed possible Flux prompt for a luxury e-commerce product photo. Format: Professional luxury e-commerce product photography of [exact garment description], [color], [material], [every visible feature]. The garment is photographed using the 'invisible mannequin' technique, giving it a perfect structured 3D stuffed shape with natural volume as if being worn, but with no body or hanger visible. Perfectly pressed, no wrinkles, crisp edges. Clean Massimo Dutti aesthetic. Symmetrically arranged, centered. Pure bright white seamless background (#FFFFFF), professional studio softbox lighting, ultra realistic, 8k, sharp focus, highly detailed texture, high end catalog presentation."
}`
            }
          ]
        }]
      })
    });

    const data = await response.json();
    if (data.error) return res.status(400).json({ error: data.error.message || "Gemini API Error" });

    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const clean = text.replace(/```json|```/g, "").trim();
    return res.status(200).json(JSON.parse(clean));
  } catch (e) {
    console.error("AI Studio Analyze Error:", e);
    return res.status(500).json({ error: e.message || "Internal server error" });
  }
});

/**
 * POST /api/studio/generate
 * Generates a high-quality product shot using Flux-dev via Replicate.
 */
router.post("/generate", async (req, res) => {
  const { prompt } = req.body;
  const REPLICATE_KEY = process.env.REPLICATE_API_TOKEN || process.env.REPLICATE_API_KEY;

  if (!REPLICATE_KEY) return res.status(500).json({ error: "REPLICATE_API_TOKEN not set" });
  if (!prompt) return res.status(400).json({ error: "No prompt provided" });

  try {
    // Create prediction with flux-dev for higher quality
    const createRes = await fetch("https://api.replicate.com/v1/models/black-forest-labs/flux-dev/predictions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${REPLICATE_KEY}`,
        "Prefer": "wait"
      },
      body: JSON.stringify({
        input: {
          prompt,
          aspect_ratio: "3:4",
          num_outputs: 1,
          output_format: "png",
          output_quality: 100,
          num_inference_steps: 35
        }
      })
    });

    if (!createRes.ok) {
      const err = await createRes.json().catch(() => ({}));
      return res.status(400).json({ error: err.detail || err.title || `Replicate error ${createRes.status}` });
    }

    const prediction = await createRes.json();

    if (prediction.status === "succeeded" && prediction.output) {
      // Replicate might return array or single URL
      const imageUrl = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;
      // Fetch the image and return as base64 so client doesn't need to hit Replicate directly
      const imgRes = await fetch(imageUrl);
      const buffer = await imgRes.arrayBuffer();
      const base64 = Buffer.from(buffer).toString("base64");
      const contentType = imgRes.headers.get("content-type") || "image/webp";
      return res.status(200).json({ image: `data:${contentType};base64,${base64}` });
    }

    // Polling if 'Prefer: wait' timed out and it's still processing
    const pollUrl = prediction.urls?.get || `https://api.replicate.com/v1/predictions/${prediction.id}`;
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const pollRes = await fetch(pollUrl, {
        headers: { "Authorization": `Bearer ${REPLICATE_KEY}` }
      });
      const poll = await pollRes.json();
      if (poll.status === "succeeded") {
        const imageUrl = Array.isArray(poll.output) ? poll.output[0] : poll.output;
        const imgRes = await fetch(imageUrl);
        const buffer = await imgRes.arrayBuffer();
        const base64 = Buffer.from(buffer).toString("base64");
        const contentType = imgRes.headers.get("content-type") || "image/webp";
        return res.status(200).json({ image: `data:${contentType};base64,${base64}` });
      }
      if (poll.status === "failed" || poll.status === "canceled") {
        return res.status(400).json({ error: poll.error || "Generation failed" });
      }
    }
    return res.status(408).json({ error: "Generation timed out" });
  } catch (e) {
    console.error("AI Studio Generate Error:", e);
    return res.status(500).json({ error: e.message || "Internal server error" });
  }
});

export default router;
