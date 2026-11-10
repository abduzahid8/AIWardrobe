/**
 * Hosted whole-outfit visual critic for the Stylist 6 offline pilot.
 *
 * REAL implementation (not a mock): sends the authorized reference photo
 * plus the actual candidate product photos to a hosted multimodal model
 * (Gemini `generateContent`) and returns structured whole-outfit verdicts in
 * the exact contract `backend/stylist6/prototype.mjs` consumes.
 *
 * Fail-closed rules (no silent fake quality):
 * - No API key  -> the factory throws; the pilot records visual verdicts
 *   as NOT RUN instead of substituting text-only scores.
 * - Reference bytes unresolvable -> every candidate abstains. A
 *   reference-vs-outfit comparison without the reference is not performed.
 * - Any product photo unfetchable, any row unparseable or out of range ->
 *   that candidate abstains (it is omitted from the returned array, so the
 *   engine can never rank it as visually approved).
 * - No training, no persistence, no quota mutation. Bounded image bytes,
 *   bounded timeout, bounded concurrency of 1.
 */

export const HOSTED_VISUAL_CRITIC_VERSION = 'hosted-visual-critic-v1';

const DEFAULT_MODEL = 'gemini-2.0-flash';
const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024;

const isFinite01 = (v) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;

function validRow(row, allowedIds) {
  if (!row || typeof row !== 'object' || !allowedIds.has(row.candidateId)) return false;
  if (!isFinite01(row.visualCohesion) || !isFinite01(row.styleFit) ||
      !isFinite01(row.occasionFit) || !isFinite01(row.formalityFit) ||
      !isFinite01(row.confidence)) return false;
  if (!Array.isArray(row.evidence) || row.evidence.length < 3 ||
      !row.evidence.every((e) => typeof e === 'string' && e.length > 0)) return false;
  if (row.reject !== undefined && typeof row.reject !== 'boolean') return false;
  return true;
}

function extractJson(text) {
  const match = String(text || '').match(/\{[\s\S]*\}|\[[\s\S]*\]/);
  if (!match) throw new Error('critic_unparseable_response');
  return JSON.parse(match[0]);
}

function buildPrompt(candidate, request) {
  const products = candidate.products.map((p) => `- ${p.id}: ${p.name || p.id}`).join('\n');
  return [
    'You are a menswear stylist judging ONE candidate outfit against a reference look.',
    'The FIRST image is the authorized reference full-outfit photo.',
    `The following ${candidate.products.length} images are the candidate real product photos in slot order:`,
    products,
    `Requested style: ${request?.style || 'unspecified'}. Requested occasion: ${request?.occasion || 'unspecified'}.`,
    'Judge the candidate as a WHOLE OUTFIT: do the real garments cohere with each other and with the reference',
    'register (formality, palette, silhouette, seasonal weight)? Penalize clashing formality, clashing',
    'patterns, or a layer combination that makes no temperature sense.',
    'Return ONLY JSON with this exact shape:',
    '{"candidateId":"<id>","visualCohesion":0-1,"styleFit":0-1,"occasionFit":0-1,',
    '"formalityFit":0-1,"confidence":0-1,"reject":false,',
    '"evidence":["<specific garment-level observation>", "...at least 3 entries..."]}',
    `candidateId must be exactly "${candidate.id}". No markdown, no extra keys.`,
  ].join('\n');
}

async function fetchImageBytes(fetchImpl, url, maxBytes, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, {
      signal: controller.signal,
      headers: { 'User-Agent': 'AIWardrobe-Stylist6-Pilot/1.0' },
    });
    if (!response.ok) throw new Error(`image_http_${response.status}`);
    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.length < 1024 || buffer.length > maxBytes) throw new Error('image_size_invalid');
    const jpeg = buffer[0] === 0xff && buffer[1] === 0xd8;
    const png = buffer[0] === 0x89 && buffer[1] === 0x50;
    if (!jpeg && !png) throw new Error('image_not_photo');
    return { base64: buffer.toString('base64'), mimeType: jpeg ? 'image/jpeg' : 'image/png' };
  } finally {
    clearTimeout(timer);
  }
}

export function createHostedVisualCritic({
  apiKey,
  model = DEFAULT_MODEL,
  endpoint = 'https://generativelanguage.googleapis.com',
  fetchImpl = globalThis.fetch,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  maxImageBytes = DEFAULT_MAX_IMAGE_BYTES,
  estimatedCostUsdPerCall = 0.02,
  resolveReference = null,
  onEvent = null,
} = {}) {
  if (!apiKey || typeof apiKey !== 'string' || apiKey.length < 20) {
    throw new Error('hosted_visual_critic_not_configured');
  }
  if (typeof fetchImpl !== 'function') throw new Error('hosted_visual_critic_fetch_required');
  const emit = (event) => {
    if (typeof onEvent === 'function') {
      try { onEvent(event); } catch { /* telemetry must never break judging */ }
    }
  };

  /**
   * Prototype-compatible critic: (payload, request) -> critiques[].
   * Abstention (omitted candidates) is the only failure mode — a failed
   * judgment never becomes a mid-range score.
   */
  return async function hostedVisualCritic(payload, request) {
    const allowed = new Set((payload || []).map((c) => c.id));
    const critiques = [];
    for (const candidate of payload || []) {
      const started = Date.now();
      try {
        const reference = typeof resolveReference === 'function'
          ? await resolveReference(candidate.referenceId)
          : null;
        if (!reference || !reference.base64) {
          emit({ candidateId: candidate.id, outcome: 'abstain', reason: 'reference_unresolved' });
          continue;
        }
        const parts = [{ text: buildPrompt(candidate, request) }];
        parts.push({ inlineData: { mimeType: reference.mimeType || 'image/jpeg', data: reference.base64 } });
        for (const product of candidate.products || []) {
          const photo = await fetchImageBytes(fetchImpl, product.imageUrl, maxImageBytes, timeoutMs);
          parts.push({ inlineData: { mimeType: photo.mimeType, data: photo.base64 } });
        }
        const url = `${endpoint.replace(/\/+$/, '')}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        let data;
        try {
          const response = await fetchImpl(url, {
            method: 'POST',
            signal: controller.signal,
            headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
            body: JSON.stringify({
              contents: [{ role: 'user', parts }],
              generationConfig: { responseMimeType: 'application/json', maxOutputTokens: 600, temperature: 0.2 },
            }),
          });
          if (!response.ok) throw new Error(`provider_http_${response.status}`);
          data = await response.json();
        } finally {
          clearTimeout(timer);
        }
        const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text || '').join('') || '';
        const row = extractJson(text);
        if (row.candidateId !== candidate.id) throw new Error('critic_candidate_mismatch');
        if (!validRow(row, allowed)) throw new Error('critic_row_invalid');
        critiques.push({
          candidateId: row.candidateId,
          visualCohesion: row.visualCohesion,
          styleFit: row.styleFit,
          occasionFit: row.occasionFit,
          formalityFit: row.formalityFit,
          confidence: row.confidence,
          evidence: row.evidence,
          ...(row.reject === true ? { reject: true } : {}),
          criticVersion: `${HOSTED_VISUAL_CRITIC_VERSION}/${model}`,
          latencyMs: Date.now() - started,
          estimatedCostUsd: estimatedCostUsdPerCall,
        });
        emit({ candidateId: candidate.id, outcome: 'judged', latencyMs: Date.now() - started });
      } catch (error) {
        emit({
          candidateId: candidate.id,
          outcome: 'abstain',
          reason: error instanceof Error ? error.message : 'unknown',
        });
      }
    }
    return critiques;
  };
}
