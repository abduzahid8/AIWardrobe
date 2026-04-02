export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const { image, mediaType } = req.body;
  const GEMINI_KEY = process.env.GEMINI_API_KEY;

  if (!GEMINI_KEY) return res.status(500).json({ error: "GEMINI_API_KEY not set" });

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
            { inlineData: { mimeType: mediaType, data: image } },
            {
              text: `You are an expert fashion stylist and luxury e-commerce product photographer for brands like Massimo Dutti.
Analyze this clothing photo carefully. Focus on the main garment only.

Respond ONLY with a JSON object containing these specific fields:
{
  "garmentType": "specific garment name",
  "color": "precise color with hex if possible",
  "material": "fabric/material description",
  "details": "key design details: collar, buttons, pockets, zippers, stitching, fit",
  "category": "top / bottom / dress / outerwear / other",
  "generationPrompt": "Write the most detailed possible Flux prompt for a luxury e-commerce product photo. Format: Professional luxury e-commerce product photography of [exact garment description], [color], [material], [every visible feature]. The garment is photographed using the 'invisible mannequin' technique, giving it a perfect structured 3D stuffed shape with natural volume as if being worn, but with no body or hanger visible. Perfectly pressed, no wrinkles, crisp edges. Clean Massimo Dutti aesthetic. Symmetrically arranged, centered. Pure bright white seamless background (#FFFFFF), professional studio softbox lighting, ultra realistic, 8k, sharp focus, highly detailed texture, high end catalog presentation."
}`
            }
          ]
        }]
      })
    });

    const data = await response.json();
    if (data.error) return res.status(400).json({ error: data.error.message });

    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const clean = text.replace(/```json|```/g, "").trim();
    return res.status(200).json(JSON.parse(clean));
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
}
