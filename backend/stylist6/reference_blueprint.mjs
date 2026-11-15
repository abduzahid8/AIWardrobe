/**
 * Human-curated, repository-owned *textual* fashion briefs.
 * No reference photo, upload, asset URL, or license claim is carried to the
 * external visual model. Photographic reference fidelity stays unavailable
 * until exact source bytes AND inference rights have been verified.
 */
export const REFERENCE_BLUEPRINT_VERSION = 's6-reference-blueprint-v1';
const ID = /^rt_\d{3}$/;
const safe = (v, max = 80) => typeof v === 'string'
  ? v.replace(/[\r\n\t\x00-\x1f]/g, ' ').replace(/\s+/g, ' ').slice(0, max) : '';
const slots = ['top','outerwear','bottom','shoes'];
export function buildReferenceBlueprints(index) {
  if (index?.source !== 'user_uploaded_flat_lay_reference_archive' || !Array.isArray(index.images))
    throw new Error('invalid_reference_blueprint_index');
  const map = new Map();
  for (const row of index.images) {
    if (!ID.test(row?.id || '') || row?.complete_outfit_collage !== true ||
        !Array.isArray(row.palette) || !row.garments || map.has(row.id)) continue;
    map.set(row.id, Object.freeze({
      id: row.id,
      name: safe(row.title, 90),
      palette: row.palette.filter(x=>typeof x==='string').slice(0,6).map(x=>safe(x,24)),
      silhouette: safe(row.silhouette, 60),
      layerPolicy: ['required','absent'].includes(row.layer_policy) ? row.layer_policy : 'unknown',
      garments: Object.fromEntries(slots.map(slot=>[slot,safe(row.garments[slot],90)])),
      // This is a *description of* a user-supplied collage, never a license
      // or proof that the actual collage is in the model's visual context.
      evidenceMode: 'textual_editorial_brief_not_visual_reference',
    }));
  }
  return map;
}
