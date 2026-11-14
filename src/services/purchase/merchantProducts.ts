import { parseIncomingProduct } from './productInput';
import { adaptMerchantMeasurements, type MerchantMeasurementChart } from './physicalDataAdapter';
import type { ShopCatalogItem } from '../../types/shop';
import type { MerchantOffer } from '../../types/purchase';
import type { GarmentPhysicalProfile } from '../../types/garment';
import {
  familyIdentityCandidate,
  canonicalCatalogProductUrl,
  trustedMerchantModelReference,
} from '../../utils/catalogIdentity';
import { inferShopRowMacro } from '../../utils/shopCatalogOrder';
export interface MerchantSnapshot {
  url: string;
  html?: string;
  structuredData?: unknown;
  observedAt: string;
  trustedSizeCharts?: GarmentPhysicalProfile[];
  verifiedMeasurementChart?: MerchantMeasurementChart;
}
export interface MerchantProduct {
  candidate?: ShopCatalogItem;
  variants: ShopCatalogItem[];
  offers: MerchantOffer[];
  sizeCharts: GarmentPhysicalProfile[];
  confidence: 'low' | 'medium' | 'high';
  evidence: string[];
}
export interface MerchantProductAdapter {
  id: string;
  canHandle: (url: URL) => boolean;
  normalizeUrl: (url: URL) => string;
  extractStableProductReference: (url: URL) => string | undefined;
  parseProduct: (snapshot: MerchantSnapshot) => MerchantProduct | null;
  parseVariants: (snapshot: MerchantSnapshot) => ShopCatalogItem[];
  parseSizeChart: (snapshot: MerchantSnapshot) => GarmentPhysicalProfile[];
  parseOffers: (snapshot: MerchantSnapshot) => MerchantOffer[];
}
export { parseIncomingProduct } from './productInput';

function productNodes(snapshot: MerchantSnapshot): Record<string, any>[] {
  const values: unknown[] = [];
  if (snapshot.structuredData) values.push(snapshot.structuredData);
  for (const match of (snapshot.html || '')
    .slice(0, 250000)
    .matchAll(/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    if (values.length >= 8) break;
    try {
      values.push(JSON.parse(match[1].slice(0, 80000)));
    } catch {
      /* Incomplete markup is expected; never evaluate scripts. */
    }
  }
  const nodes: Record<string, any>[] = [];
  const walk = (value: unknown, depth = 0) => {
    if (depth > 5 || nodes.length >= 50) return;
    if (Array.isArray(value)) {
      value.slice(0, 30).forEach((v) => walk(v, depth + 1));
      return;
    }
    if (!value || typeof value !== 'object') return;
    const row = value as Record<string, any>;
    if (
      ['Product', 'ProductGroup'].some((type) =>
        Array.isArray(row['@type']) ? row['@type'].includes(type) : row['@type'] === type
      )
    )
      nodes.push(row);
    if (row['@graph']) walk(row['@graph'], depth + 1);
    if (row.hasVariant) walk(row.hasVariant, depth + 1);
  };
  values.forEach((value) => walk(value));
  return nodes;
}
function parsedProduct(
  snapshot: MerchantSnapshot,
  id: string,
  known: boolean
): MerchantProduct | null {
  const input = parseIncomingProduct(snapshot.url);
  if (!input.url) return null;
  const url = new URL(input.url),
    nodes = productNodes(snapshot);
  if (!nodes.length) return null;
  const product = nodes.find((node) => node.hasVariant) || nodes[0];
  const childRows = product.hasVariant
    ? (Array.isArray(product.hasVariant) ? product.hasVariant : [product.hasVariant])
        .filter((node: unknown) => node && typeof node === 'object')
        .map(
          (node: Record<string, any>) =>
            nodes.find((actual) => node['@id'] && actual['@id'] === node['@id']) || node
        )
    : [];
  const isGroup = Array.isArray(product['@type'])
    ? product['@type'].includes('ProductGroup')
    : product['@type'] === 'ProductGroup';
  const selectable: Record<string, any>[] = childRows.length ? childRows : isGroup ? [] : [product];
  if (!selectable.length) return null;
  if (typeof product.name !== 'string' || !product.name.trim()) return null;
  const make = (row: Record<string, any>, index: number): ShopCatalogItem => {
    const rawOffer = Array.isArray(row.offers) ? row.offers[0] : row.offers;
    const price = Number(rawOffer?.price);
    const name = String(row.name || product.name).slice(0, 200),
      macro = inferShopRowMacro(String(row.category || ''), name);
    const image = Array.isArray(row.image) ? row.image[0] : row.image;
    const suppliedLink = parseIncomingProduct(String(row.url || rawOffer?.url || ''));
    const ownLink =
      suppliedLink.url && new URL(suppliedLink.url).hostname === url.hostname
        ? suppliedLink.url
        : undefined;
    const brand = row.brand || product.brand;
    return {
      id: `url:${id}:${String(row.sku || row.productID || index).slice(0, 100)}`,
      name,
      brand: typeof brand === 'string' ? brand : String(brand?.name || ''),
      price: Number.isFinite(price) && price > 0 ? price : 0,
      currency:
        typeof rawOffer?.priceCurrency === 'string' &&
        /^[A-Z]{3}$/.test(rawOffer.priceCurrency.toUpperCase())
          ? rawOffer.priceCurrency.toUpperCase()
          : undefined,
      imageUrl: typeof image === 'string' && /^https?:\/\//i.test(image) ? image : '',
      garmentType:
        /\b(dress|dresses|gown|jumpsuit)\b/i.test(String(row.category || '')) ||
        (/\b(dress|gown|jumpsuit)\b/i.test(name) &&
          !/\bdress\s+(shirt|shoes|trousers|suit)\b/i.test(name))
          ? 'dresses'
          : macro === 'other'
            ? 'accessory'
            : macro === 'shoes'
              ? 'shoes'
              : macro === 'bottom'
                ? 'lower_body'
                : 'upper_body',
      category: macro,
      description: typeof row.description === 'string' ? row.description.slice(0, 2000) : undefined,
      source: 'shared-url-unverified',
      sourceUrl: ownLink || input.url,
      colorName: typeof row.color === 'string' ? row.color : undefined,
      availableSizes: typeof row.size === 'string' ? [row.size] : undefined,
    };
  };
  const variants = selectable.slice(0, 20).map(make);
  const explicitMatches = selectable
    .slice(0, 20)
    .map((node, index) => ({ node, index }))
    .filter(
      ({ node }) =>
        parseIncomingProduct(
          String(
            node.url || (Array.isArray(node.offers) ? node.offers[0]?.url : node.offers?.url) || ''
          )
        ).url === input.url
    );
  const candidate =
    variants.length === 1
      ? variants[0]
      : explicitMatches.length === 1
        ? variants[explicitMatches[0].index]
        : undefined;
  const offers: MerchantOffer[] = [];
  selectable.slice(0, 20).forEach((node, index) => {
    const rows = Array.isArray(node.offers) ? node.offers : [node.offers];
    for (const offer of rows.slice(0, 10)) {
      if (
        !Number.isFinite(Date.parse(snapshot.observedAt)) ||
        !offer ||
        typeof offer !== 'object' ||
        !Number.isFinite(Number(offer.price)) ||
        Number(offer.price) <= 0 ||
        typeof offer.priceCurrency !== 'string' ||
        !/^[A-Z]{3}$/.test(offer.priceCurrency.toUpperCase())
      )
        continue;
      const link = parseIncomingProduct(
        String(offer.url || variants[index].sourceUrl || input.url)
      );
      if (!link.url || new URL(link.url).hostname !== url.hostname) continue;
      const gtin = String(node.gtin || node.gtin13 || node.gtin14 || node.gtin12 || '');
      const stock =
        typeof offer.availability === 'string'
          ? /\/InStock$/.test(offer.availability)
            ? true
            : /\/(OutOfStock|Discontinued)$/.test(offer.availability)
              ? false
              : undefined
          : undefined;
      offers.push({
        id: `offer:${id}:${String(node.sku || index)}`,
        variantId: variants[index].id,
        merchant: url.hostname,
        merchantProductId: node.sku,
        url: link.url,
        currency: offer.priceCurrency.toUpperCase(),
        currentPrice: Number(offer.price),
        inStock: stock,
        sizes:
          typeof node.size === 'string' && stock !== undefined ? { [node.size]: stock } : undefined,
        lastSeenAt: snapshot.observedAt,
        merchantConfidence: known ? 'medium' : 'low',
        ...(gtin ? { canonicalIdentifier: { kind: 'gtin' as const, value: gtin } } : {}),
      });
    }
  });
  const measured =
    known && snapshot.verifiedMeasurementChart
      ? variants.flatMap((variant, index) =>
          adaptMerchantMeasurements(snapshot.verifiedMeasurementChart!, {
            id: variant.id,
            reference: String(selectable[index].sku || selectable[index].productID || ''),
            url: url.toString(),
          })
        )
      : [];
  for (const variant of variants) {
    const profiles = measured.filter((profile) => profile.garmentId === variant.id);
    if (profiles.length) {
      variant.physicalProfiles = profiles;
      variant.availableSizes = profiles.map((profile) => profile.sizeLabel);
    }
  }
  return {
    candidate,
    variants,
    offers,
    sizeCharts: known
      ? [...(snapshot.trustedSizeCharts || []), ...measured]
          .filter((p) => variants.some((v) => v.id === p.garmentId))
          .slice(0, 30)
      : [],
    confidence: known ? 'medium' : 'low',
    evidence: [
      'Parsed supplied structured product data without fetching or executing merchant code',
      ...(known
        ? ['Recognized merchant adapter']
        : ['Unrecognized merchant; metadata needs confirmation']),
    ],
  };
}
const merchants: Record<string, string[]> = {
  zara: ['zara.com'],
  uniqlo: ['uniqlo.com'],
  cos: ['cos.com', 'cosstores.com'],
  massimo: ['massimodutti.com'],
  jackjones: ['jackjones.com'],
  jcrew: ['jcrew.com'],
};
export const merchantAdapters: MerchantProductAdapter[] = Object.entries(merchants).map(
  ([id, hosts]) => ({
    id,
    canHandle: (url) =>
      hosts.some((host) => url.hostname === host || url.hostname === `www.${host}`),
    normalizeUrl: (url) => parseIncomingProduct(url.toString()).url!,
    extractStableProductReference: (url) => {
      const value = url.pathname.match(/-p(\d+)\.html|\/(?:products?|product|p)\/([^/]+)/i);
      return value?.[1] || value?.[2];
    },
    parseProduct: (snapshot) => parsedProduct(snapshot, id, true),
    parseVariants: (snapshot) => parsedProduct(snapshot, id, true)?.variants || [],
    parseOffers: (snapshot) => parsedProduct(snapshot, id, true)?.offers || [],
    parseSizeChart: (snapshot) => parsedProduct(snapshot, id, true)?.sizeCharts || [],
  })
);
export interface ResolvedPurchaseInput {
  status: 'exact' | 'family' | 'parsed' | 'partial' | 'unresolved';
  candidate?: ShopCatalogItem;
  variants: ShopCatalogItem[];
  offers: MerchantOffer[];
  evidence: string[];
  message?: string;
  url?: string;
  title?: string;
}
export async function resolvePurchaseInput(
  input: string,
  options: {
    catalog?: readonly ShopCatalogItem[];
    snapshot?: MerchantSnapshot;
    lookupUrl?: (url: string) => Promise<ShopCatalogItem[]>;
  } = {}
): Promise<ResolvedPurchaseInput> {
  const incoming = parseIncomingProduct(input);
  if (incoming.error)
    return {
      status: 'unresolved',
      variants: [],
      offers: [],
      evidence: [],
      message: incoming.error,
    };
  if (!incoming.url)
    return {
      status: 'partial',
      title: incoming.title,
      variants: [],
      offers: [],
      evidence: ['Title only; select a matching catalog product or provide product details'],
    };
  let catalog = [...(options.catalog || [])].slice(0, 500);
  const normalized = (url?: string) => (url ? parseIncomingProduct(url).url : undefined);
  let exact = catalog.find((item) => normalized(item.sourceUrl) === incoming.url);
  if (!exact && options.lookupUrl) {
    try {
      const rows = await options.lookupUrl(incoming.url);
      catalog = [...catalog, ...rows.slice(0, 20)];
      exact = rows.find((item) => normalized(item.sourceUrl) === incoming.url);
    } catch {
      /* Offline resolution retains the local receiving flow. */
    }
  }
  if (exact)
    return {
      status: 'exact',
      url: incoming.url,
      candidate: exact,
      variants: [exact],
      offers: [],
      evidence: ['Exact existing catalog URL; original row ID retained'],
    };
  const url = new URL(incoming.url),
    adapter = merchantAdapters.find((a) => a.canHandle(url));
  const canonical = canonicalCatalogProductUrl(incoming.url);
  const family = catalog.filter(
    (item) =>
      (canonical && canonicalCatalogProductUrl(item.sourceUrl) === canonical) ||
      (!!adapter &&
        trustedMerchantModelReference(item) &&
        trustedMerchantModelReference({ ...item, sourceUrl: incoming.url }) ===
          trustedMerchantModelReference(item))
  );
  if (family.length)
    return {
      status: 'family',
      url: incoming.url,
      candidate: undefined,
      variants: family,
      offers: [],
      evidence: ['High-confidence model relationship; select an existing concrete variant'],
      message: 'Choose a catalog variant. A different URL color is not silently substituted.',
    };
  const snapshot =
    options.snapshot && normalized(options.snapshot.url) === incoming.url
      ? options.snapshot
      : undefined;
  const parsed = snapshot
    ? adapter?.parseProduct(snapshot) || parsedProduct(snapshot, url.hostname, false)
    : null;
  if (parsed)
    return {
      status: parsed.candidate ? 'parsed' : 'family',
      url: incoming.url,
      ...parsed,
      evidence: parsed.evidence,
      ...(!parsed.candidate
        ? {
            message:
              'Choose the concrete variant; a product group is not a selectable physical item',
          }
        : {}),
    };
  return {
    status: 'partial',
    url: incoming.url,
    variants: [],
    offers: [],
    evidence: [
      adapter
        ? 'Supported merchant URL; no product snapshot available offline'
        : 'Unknown merchant; automatic scraping is not enabled',
    ],
    message:
      'Select a catalog product or enter its title, brand, size and native price to continue',
  };
}
