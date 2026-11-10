/**
 * Read-only, fail-closed interface between owner-authorized image library and
 * a Reference-First Stylist 6 experiment. This intentionally does NOT score
 * fashion quality, infer image tags or activate unreviewed training images.
 *
 * The actual image matching implementation must compare authorized *outfit*
 * reference evidence with catalog items and be validated against human style
 * judgment. No production route imports this file.
 */
export const REFERENCE_LIBRARY_VERSION = "reference-library-v1";

const tags = (values) => Array.isArray(values)
  ? [...new Set(values.map(v => String(v).trim().toLowerCase()).filter(Boolean))]
  : [];

export function acceptedReferences(manifest, filters = {}) {
  if (!manifest || manifest.schema_version !== 1 ||
      manifest.source !== "instagram_creator_export" ||
      typeof manifest.account !== "string" || !Array.isArray(manifest.assets)) {
    throw new Error("invalid_creator_reference_manifest");
  }
  if (filters.account && manifest.account !== filters.account) {
    throw new Error("creator_account_mismatch");
  }
  return manifest.assets
    .filter(a =>
      a &&
      a.source_account === manifest.account &&
      a.review_status === "approved_for_reference_use" &&
      a.contains_full_outfit === true &&
      a.rights_confirmed === true &&
      typeof a.sha256 === "string" &&
      /^[0-9a-f]{64}$/.test(a.sha256) &&
      typeof a.local_path === "string" &&
      /^images\/[0-9a-f]{64}\.(jpg|jpeg|png|webp|heic|heif)$/.test(a.local_path) &&
      tags(a.style_tags).length > 0 &&
      tags(a.occasion_tags).length > 0 &&
      Number.isFinite(a.quality_score) &&
      a.quality_score >= 0.7 &&
      a.quality_score <= 1
    )
    .filter(a => !filters.style || tags(a.style_tags).includes(String(filters.style).toLowerCase()))
    .filter(a => !filters.occasion ||
      tags(a.occasion_tags).includes(String(filters.occasion).toLowerCase()))
    .map(a => Object.freeze({
      sourceAccount: manifest.account,
      sha256: a.sha256,
      path: a.local_path,
      sourcePostUrl: a.source_post_url || null,
      styleTags: tags(a.style_tags),
      occasionTags: tags(a.occasion_tags),
      quality: a.quality_score,
    }));
}
