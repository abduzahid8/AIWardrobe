/**
 * Stylist 6 — explicit dress intent without conflating the fashion term
 * "smart casual" with "relaxed casual". Keep the interpretation shared by
 * catalog discovery, weather composition and whole-outfit editorial admission.
 * All signals are derived from the bounded supplied request, never personal
 * demographic guesses or an item brand.
 */
export function deliberatelyRelaxedIntent(value) {
  if (typeof value !== 'string') return false;
  const text=value.normalize('NFKC').toLowerCase().slice(0,600)
    // A fashion dress code, not permission to wear jeans and a basic tee.
    .replace(/\b(?:smart|business)[\s_-]+casual\b/g,' ');
  return /\b(?:casual|relaxed|informal|sporty|weekend|dress[ -]?down)\b|\bповседневн|расслабленн|прогулк|выходн/.test(text);
}
