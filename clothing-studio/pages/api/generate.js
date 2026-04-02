export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const { prompt } = req.body;
  const REPLICATE_KEY = process.env.REPLICATE_API_KEY;

  if (!REPLICATE_KEY) return res.status(500).json({ error: "REPLICATE_API_KEY not set" });

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

    if (prediction.status === "succeeded") {
      const imageUrl = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;
      // Fetch the image and return as base64 so client doesn't need to hit Replicate directly
      const imgRes = await fetch(imageUrl);
      const buffer = await imgRes.arrayBuffer();
      const base64 = Buffer.from(buffer).toString("base64");
      const contentType = imgRes.headers.get("content-type") || "image/webp";
      return res.status(200).json({ image: `data:${contentType};base64,${base64}` });
    }

    // Poll if not done
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
    return res.status(500).json({ error: e.message });
  }
}

export const config = { api: { bodyParser: { sizeLimit: "10mb" }, responseLimit: "20mb" } };
