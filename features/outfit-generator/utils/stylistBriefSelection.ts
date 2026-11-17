/**
 * A chip tap is an explicit change of the currently selected style/occasion.
 * Preserve freely written constraints (budget, fabrics, body/fit preferences),
 * but remove *only* old standalone style/occasion tags from a previous brief.
 *
 * Example: "work, classic, no bright colors" -> selecting Old Money / Date
 * gives "no bright colors". Do not erase narrative phrases such as
 * "work dinner with clients" or reinterpret them as disposable tags.
 */
const STYLE_TAGS: ReadonlyArray<[string, RegExp]> = [
  ['old_money', /^(?:old[ _-]*money|quiet[ _-]*luxury)$/i],
  ['semi_classic', /^(?:semi[ _-]*classic|semi[ _-]*formal)$/i],
  ['business_casual', /^(?:business[ _-]*casual|modern[ _-]*professional)$/i],
  ['classic', /^(?:classic|classical)$/i],
  ['minimalist', /^(?:minimalist|minimalism|minimal)$/i],
  ['casual', /^casual$/i],
];
const OCCASION_TAGS: ReadonlyArray<[string, RegExp]> = [
  ['work', /^(?:work|office|business|corporate|meeting)$/i],
  ['date', /^(?:date|date[ _-]*night|romantic[ _-]*date)$/i],
  ['formal', /^(?:formal|wedding|gala|black[ _-]*tie)$/i],
  ['sport', /^(?:sport|gym|workout|training)$/i],
  ['travel', /^(?:travel|trip|airport)$/i],
  ['casual', /^(?:casual|weekend|everyday)$/i],
];
const matching = (tag: string, table: ReadonlyArray<[string, RegExp]>) =>
  table.find(([, re]) => re.test(tag))?.[0];

/** Idempotent. Treat non-tag free text and budget phrases as opaque. */
export function reconcileStylistBriefWithChips(
  brief: string,
  style: string | null | undefined,
  occasion: string | null | undefined,
): string {
  const raw = String(brief || '');
  if (!raw.trim()) return raw;
  // A missing selection does not authorize discarding the user's text.
  if (!style && !occasion) return raw;
  const parts = raw.split(/[,;\n]/);
  const kept = parts.filter(part => {
    const tag = part.trim();
    const taggedStyle = matching(tag, STYLE_TAGS);
    const taggedOccasion = matching(tag, OCCASION_TAGS);
    if (!taggedStyle && !taggedOccasion) return true;
    // "casual" can describe both style and occasion. Preserve it if
    // either selected chip still matches.
    if ((taggedStyle && taggedStyle === style) ||
        (taggedOccasion && taggedOccasion === occasion)) return true;
    return !((taggedStyle && style) || (taggedOccasion && occasion));
  });
  if (kept.length === parts.length) return raw;
  return kept.map(part => part.trim()).filter(Boolean).join(', ');
}
