import type { ShopCatalogItem } from '../types/shop';
import {
  productIdentityOf,
  type ProductIdentity,
} from '../../backend/supabase/functions/_shared/outfitDiversity';
import { garmentVarietyFamily, inferShopRowMacro } from './shopCatalogOrder';

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
