/**
 * Bounded merchant-specific provenance for physical color.
 *
 * Some FR Group product titles omit the color completely even though their
 * canonical Uzbek product URL ends with a verified transliterated color slug.
 * Fallback only when title itself does not state a color. Path only, no
 * marketing query params; strict hostname and store route; no invented hue
 * for "multicolor" or unknown colorways.
 */
const FR_COLORS = Object.freeze({
  chernyj: 'black',
  korichnevyj: 'brown',
  bezhevyj: 'beige',
  sinij: 'blue',
  seryj: 'grey',
  belyj: 'white',
  zelenyj: 'green',
});

export function frColorFromCanonicalUrl(url) {
  try {
    const value = new URL(String(url || ''));
    if (value.protocol !== 'https:' || value.hostname.toLowerCase() !== 'frgroup.uz' ||
        value.port || value.username || value.password ||
        !value.pathname.startsWith('/ru/store/')) return null;
    const productName = decodeURIComponent(value.pathname.split('/').at(-1) || '').toLowerCase();
    const suffix = productName.match(/-([a-z]+)$/)?.[1];
    return suffix && Object.hasOwn(FR_COLORS, suffix) ? FR_COLORS[suffix] : null;
  } catch {
    return null;
  }
}
