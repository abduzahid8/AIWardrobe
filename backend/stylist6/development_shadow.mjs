// Private integration boundary after Stylist 5 auth, quota, catalog retrieval,
// personalization, feasibility and selection. No HTTP route imports this module.
import { createHostedVisualCritic, MAX_VISUAL_CANDIDATES, verifiedVisualImage } from './visual_critic_hosted.mjs';
import { evaluateVisualShortlist, STYLIST6_VERSION } from './prototype.mjs';
import { outfitPurchaseEvidence, trialImageDecision } from './retail_evidence.mjs';

export async function compareAuthenticatedPlan({
  mode, plan, baselineDeck, catalog, photos, reference, rights, offers,
  purchase, scenario, apiKey, fetchImpl, onEvent, runId, claimRun, now = Date.now(),
}) {
  const withheld = (reason, diagnostics = {}) => ({ baselineDeck, visualDeck: null, reason, diagnostics });
  if (mode !== 'private-stylist6-shadow' || scenario?.synthetic !== true) return withheld('shadow_disabled');
  if (!/^[a-zA-Z0-9_-]{1,80}$/.test(runId || '') || typeof claimRun !== 'function') {
    return withheld('missing_shadow_admission');
  }
  if (!Array.isArray(plan?.candidates) || !Array.isArray(baselineDeck) ||
      !(catalog instanceof Map) || !(photos instanceof Map) || !Array.isArray(offers) || !purchase) {
    return withheld('missing_trusted_plan_inputs');
  }
  if (!fetchImpl && !apiKey?.startsWith('nvapi-')) return withheld('trial_credential_unavailable');
  if (!reference || trialImageDecision(rights, { kind: 'reference', id: reference.id, sourceUrl: reference.sourceUrl, sha256: reference.sha256 }, now) !== 'cleared') {
    return withheld('reference_rights_unverified');
  }
  try { verifiedVisualImage(reference); } catch { return withheld('reference_bytes_invalid'); }

  const candidates = new Map(plan.candidates.map((look) => [look.id, look]));
  const preferred = new Set([...baselineDeck, ...plan.candidates].map((look) => look.id));
  const eligible = [], resolved = new Map();
  for (const id of preferred) {
    const look = candidates.get(id);
    if (!look || !Array.isArray(look.items) || ![3, 4].includes(look.items.length)) continue;
    const products = look.items.map((item) => {
      const imageUrl = item.imageUrl || item.image_url;
      const existing = resolved.get(item.id);
      if (existing) return existing.imageUrl === imageUrl && item.isShopItem === true &&
        existing.row.price === item.price && existing.row.currency === item.currency ? existing : null;
      const row = catalog.get(item.id) || (item.id.startsWith('shop_') ? catalog.get(item.id.slice(5)) : null);
      const photo = row && photos.get(row.id);
      if (!photo || item.isShopItem !== true || row.is_active !== true || row.image_url !== imageUrl ||
          ![row.id, 'shop_' + row.id].includes(item.id) || row.price !== item.price || row.currency !== item.currency ||
          trialImageDecision(rights, { kind: 'product', id: row.id, sourceUrl: imageUrl, sha256: photo.sha256 }, now) !== 'cleared') {
        return null;
      }
      try { verifiedVisualImage(photo); } catch { return null; }
      const product = { row, photo, imageUrl };
      resolved.set(item.id, product);
      return product;
    });
    if (products.some((product) => !product)) continue;
    const retailItems = products.map(({ row }, index) => ({ ...row, macroCategory: look.items[index].macroCategory }));
    if (outfitPurchaseEvidence(retailItems, offers, { ...purchase, requireCatalogPriceMatch: true }, now).status !== 'verified') continue;
    const brief = { id: look.concept?.id || 'shared', name: look.concept?.title || 'Stylist look', referenceId: reference.id };
    eligible.push({ id, briefId: brief.id, brief, items: look.items });
    if (eligible.length === MAX_VISUAL_CANDIDATES) break;
  }
  if (eligible.length < 3) return withheld('insufficient_verified_candidates', { eligible: eligible.length });

  const critic = createHostedVisualCritic({
    apiKey, provider: 'nvidia', fetchImpl, onEvent,
    resolveReference: (id) => id === reference.id ? reference : null,
    resolveProduct: ({ id, imageUrl }) => resolved.get(id)?.imageUrl === imageUrl ? resolved.get(id).photo : null,
  });
  // The caller owns the existing evaluation quota and must supply an
  // atomic claim. A failed or ambiguous claim never triggers another call.
  try {
    if (await claimRun(runId) !== true) return withheld('shadow_already_claimed');
  } catch {
    return withheld('shadow_admission_failed');
  }
  const judged = await evaluateVisualShortlist(
    { generationVersion: STYLIST6_VERSION, shortlist: eligible, diagnostics: {} },
    { style: scenario.style, occasion: scenario.occasion, brief: scenario.brief, referenceId: reference.id },
    critic, { target: 3 },
  );
  if (judged.outfits.length !== 3) return withheld('no_complete_visual_deck', judged.diagnostics);
  return {
    baselineDeck,
    visualDeck: judged.outfits.map(({ id }) => candidates.get(id)),
    reason: null,
    diagnostics: judged.diagnostics,
  };
}
