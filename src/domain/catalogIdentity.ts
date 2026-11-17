import type { ShopCatalogItem } from '../types/shop.ts';
import { productIdentityOf, type ProductIdentity } from './outfitDiversity.ts';
import { garmentVarietyFamily, inferShopRowMacro } from '../utils/shopCatalogOrder.ts';

export interface CatalogListingIdentity {
  rowId: string;
  source: string;
  externalId?: string;
  sourceUrl?: string;
}
export type PhysicalProductIdentity = ProductIdentity;
export type GroupingConfidence = 'high' | 'medium' | 'low';
export interface ProductFamilyIdentity {
  familyKey: string;
  confidence: GroupingConfidence;
  evidence: string[];
}
export interface VariantIdentity {
  variantId: string;
  listingId: string;
}
export type CatalogListing = ShopCatalogItem;
export type ProductVariantIdentity = VariantIdentity;
export interface VisualFamilyIdentity {
  visualFamilyKey: string;
}

export function normalizeIdentityText(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}
export const normalizeCatalogBrand = normalizeIdentityText;
export const normalizeCatalogProductName = normalizeIdentityText;
const COLOR_ALIASES: Record<string, string> = {
  черный: 'black',
  чёрный: 'black',
  черная: 'black',
  чёрная: 'black',
  черные: 'black',
  чёрные: 'black',
  белый: 'white',
  белая: 'white',
  белые: 'white',
  синий: 'blue',
  синяя: 'blue',
  синие: 'blue',
  коричневый: 'brown',
  коричневая: 'brown',
  коричневые: 'brown',
  бежевый: 'beige',
  бежевые: 'beige',
  серый: 'grey',
  серая: 'grey',
  серые: 'grey',
  gray: 'grey',
};
export function catalogBaseName(item: ShopCatalogItem): string {
  return normalizeIdentityText(item.name)
    .split(' ')
    .filter((token) => !COLORS.has(token) && !COLOR_ALIASES[token])
    .join(' ');
}
export function listingIdentityOf(item: ShopCatalogItem): CatalogListingIdentity {
  return {
    rowId: item.id,
    source: item.source || '',
    externalId: item.externalId,
    sourceUrl: item.sourceUrl,
  };
}
/** Adapter to the existing physical identity; never replace its ANY-key semantics. */
export function physicalProductIdentityOf(item: ShopCatalogItem): PhysicalProductIdentity {
  return productIdentityOf(item);
}
export function variantIdentityOf(item: ShopCatalogItem): VariantIdentity {
  return { variantId: item.id, listingId: item.id };
}
export function visualFamilyIdentityOf(item: ShopCatalogItem): VisualFamilyIdentity {
  return {
    visualFamilyKey: garmentVarietyFamily({
      name: item.name,
      macroCategory: inferShopRowMacro(item.garmentType, item.name),
    }),
  };
}

const COLORS = new Set(
  'black white navy blue brown burgundy red green olive beige cream ecru grey gray charcoal pink purple yellow orange tan'.split(
    ' '
  )
);
export function catalogColorEvidence(item: ShopCatalogItem): string[] {
  if (item.colorName) {
    const color = normalizeIdentityText(item.colorName);
    return [COLOR_ALIASES[color] || color];
  }
  return [
    ...new Set(
      normalizeIdentityText(item.name)
        .split(' ')
        .map((token) => COLOR_ALIASES[token] || token)
        .filter((token) => COLORS.has(token))
    ),
  ];
}
export function catalogTitleColor(item: ShopCatalogItem): string | undefined {
  const colors = catalogColorEvidence(item);
  return colors.length === 1 ? colors[0] : undefined;
}

const GENERIC = new Set(
  'regular relaxed slim fit cotton linen wool leather suede organic stretch shirt shirts overshirt overshirts sweatshirt sweatshirts cardigan cardigans sweater jumper jacket coat blazer blazer regular tailored knitted knit ribbed textured relaxed slim straight wide leg loose washed brushed organic stretch gabardine double pleated broken in ultimate non iron chino chinos trousers pants shoes loafer loafers penny blazer oxford polo t tee crew neck long short sleeve sleeves soft classic basic premium men mens with and the a'.split(
    ' '
  )
);

/** Merchant-scoped reference adapter: the official Jack & Jones page exposes
 * the eight-digit Article number separately from the URL's color suffix.
 * Do not generalize this suffix rule to unknown feeds or arbitrary SKU strings.
 */
export function trustedMerchantModelReference(item: ShopCatalogItem): string | undefined {
  try {
    const url = new URL(item.sourceUrl || '');
    if (!['http:', 'https:'].includes(url.protocol)) return undefined;
    if (
      ['jack and jones', 'jack jones'].includes(normalizeCatalogBrand(item.brand)) &&
      /^(?:www\.)?jackjones\.com$/i.test(url.hostname)
    ) {
      const match = url.pathname.match(/^\/[a-z]{2}-[a-z]{2}\/product\/(\d{8})_\d+(?:\/|$)/i);
      if (match) return `jackjones:article:${match[1]}`;
    }
  } catch {
    /* Missing/invalid merchant URLs supply no evidence. */
  }
  return undefined;
}

/** Only an actual product route is evidence. Unknown query keys remain identity-bearing. */
export function canonicalCatalogProductUrl(raw?: string): string | undefined {
  try {
    const url = new URL(raw || '');
    if (
      !['http:', 'https:'].includes(url.protocol) ||
      !/(?:\/products?\/[^/]+|\/p\/[^/]+|-p\d+\.html)/i.test(url.pathname)
    )
      return;
    for (const key of [...url.searchParams.keys()]) {
      if (/^(?:utm_.*|color|colour|color_name|size|variant|variant_id|v1|v2|ajax|view)$/i.test(key))
        url.searchParams.delete(key);
    }
    url.hash = '';
    url.searchParams.sort();
    return `${url.hostname.toLowerCase()}${url.pathname.replace(/\/+$/, '')}${url.search}`;
  } catch {
    return undefined;
  }
}

export function familyIdentityCandidate(item: ShopCatalogItem): ProductFamilyIdentity {
  const scope = [normalizeIdentityText(item.brand), item.garmentType];
  if (!scope[0] || item.garmentType === 'outfit')
    return {
      familyKey: JSON.stringify(['listing', item.id]),
      confidence: 'low',
      evidence: ['insufficient model evidence'],
    };
  const source = item.source?.trim();
  if (source && item.sourceProductGroupId)
    return {
      familyKey: JSON.stringify([...scope, source, 'group', item.sourceProductGroupId]),
      confidence: 'high',
      evidence: ['trusted source product group'],
    };
  if (source && item.modelReference)
    return {
      familyKey: JSON.stringify([...scope, source, 'model', item.modelReference]),
      confidence: 'high',
      evidence: ['trusted model reference'],
    };
  const merchantReference = trustedMerchantModelReference(item);
  if (merchantReference)
    return {
      familyKey: JSON.stringify([...scope, 'merchant-model', merchantReference]),
      confidence: 'high',
      evidence: ['verified merchant article reference in product URL'],
    };
  const url = canonicalCatalogProductUrl(item.sourceUrl);
  if (url)
    return {
      familyKey: JSON.stringify([...scope, 'url', url]),
      confidence: 'high',
      evidence: ['canonical product URL'],
    };
  const title = catalogBaseName(item);
  if (catalogTitleColor(item) && title.split(' ').some((t) => !GENERIC.has(t) && t.length > 2)) {
    return {
      familyKey: JSON.stringify([...scope, source || '', 'title', title, item.currency || '']),
      confidence: 'medium',
      evidence: ['exact distinctive title; color evidence (requires compatible variants)'],
    };
  }
  return {
    familyKey: JSON.stringify(['listing', item.id]),
    confidence: 'low',
    evidence: ['insufficient model evidence'],
  };
}

/**
 * Repair the production Zara construction `.jpg&w=1024`: a query
 * continuation glued to the extension without `?`. Only the separator is
 * changed; host, path and query evidence are preserved. Verified in
 * production: the repaired URL resizes (~61 KB) while the malformed form
 * silently serves the full original (~242 KB). Returns undefined for a blank
 * value so callers can distinguish a missing image from a malformed one.
 */
export function normalizeCatalogImageUrl(url: unknown): string | undefined {
  const value = typeof url === 'string' ? url.trim() : '';
  if (!value) return undefined;
  // A URL that already has a query separator is not the known malformed
  // construction; repairing an `&param` inside an existing query would create
  // a second `?` and corrupt a valid URL.
  if (value.includes('?')) return value;
  return value.replace(/(\.(?:jpe?g|png|webp|avif|gif))&(?=[^&=]+=)/i, '$1?');
}

export type CatalogImageAdmission =
  | { ok: true; url: string; repaired: boolean }
  | { ok: false; reason: 'image_missing' | 'image_malformed' };

/**
 * Structural admission for catalog image URLs. The 961 production
 * `zara-uz-white-bg` rows are repaired rather than excluded: their CDN serves
 * a usable image, and dropping the whole Zara source would shrink coverage.
 * Only structurally unusable URLs (blank, control characters, non-HTTP
 * schemes, a still-glued query continuation, or multiple query separators)
 * are refused, and those are reported as `image_missing`/`image_malformed`.
 */
export function catalogImageAdmission(url: unknown): CatalogImageAdmission {
  const value = typeof url === 'string' ? url.trim() : '';
  if (!value) return { ok: false, reason: 'image_missing' };
  if ([...value].some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127 || /\s/.test(character)))
    return { ok: false, reason: 'image_malformed' };
  const repaired = normalizeCatalogImageUrl(value);
  if (!repaired) return { ok: false, reason: 'image_missing' };
  // A continuation that could not be repaired remains malformed. Only the
  // path portion is inspected: `&` inside a valid query value is data.
  const path = repaired.split('?')[0];
  if (/\.(?:jpe?g|png|webp|avif|gif)&/i.test(path))
    return { ok: false, reason: 'image_malformed' };
  if ((repaired.match(/\?/g) || []).length > 1) return { ok: false, reason: 'image_malformed' };
  try {
    const parsed = new URL(repaired);
    if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname)
      return { ok: false, reason: 'image_malformed' };
  } catch {
    // Relative paths cannot render outside their originating catalog site.
    return { ok: false, reason: 'image_malformed' };
  }
  return { ok: true, url: repaired, repaired: repaired !== value };
}

/** True when a catalog row's image URL is usable (after safe normalization). */
export function hasAdmissibleCatalogImage(row: { image_url?: unknown; imageUrl?: unknown } | null | undefined): boolean {
  if (!row) return false;
  return catalogImageAdmission(row.image_url ?? row.imageUrl).ok;
}
