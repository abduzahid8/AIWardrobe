/**
 * providers.ts — Provider-neutral vision interfaces (4D, 4Y, §48).
 *
 * No provider is hardwired: `GarmentVisionProvider` is injectable, the
 * default is a deterministic structured-metadata fallback that never
 * invents claims. Remote calls must route through the secured Edge
 * Function proxy (see aiVisionService); no keys live client-side.
 * SSRF/size/timeout guards live here for future product-image fetching.
 */
import type {
  GarmentRegion,
  GarmentVisualParse,
  OutfitVisualParse,
  VisualAsset,
} from '../../types/visualOS';

export interface VisionContext {
  userId: string;
  occasion?: string;
  knownCategory?: string;
  knownBrand?: string;
  qualityDegraded?: boolean;
}

export interface GarmentVisionProvider {
  readonly name: string;
  readonly version: string;
  analyzeGarment(asset: VisualAsset, context: VisionContext): Promise<GarmentVisualParse>;
  analyzeOutfit(asset: VisualAsset, context: VisionContext): Promise<OutfitVisualParse>;
  compareGarments(a: VisualAsset, b: VisualAsset): Promise<{ score: number; notes: string[] }>;
  extractVisualFeatures(asset: VisualAsset): Promise<Record<string, string[]>>;
}

const UNKNOWN_PARSE: GarmentVisualParse = {
  assetId: '',
  garmentType: 'unknown',
  category: 'unknown',
  dominantColors: [],
  secondaryColors: [],
  pattern: 'unknown',
  silhouette: 'unknown',
  materialClass: 'unknown',
  formality: 0.5,
  styleTags: [],
  visibleDetails: [],
  conditionHint: 'unknown',
  confidencePerField: {},
  provenance: 'deterministic-fallback',
  parsedAt: '',
};

/**
 * Deterministic fallback: emits only what structured metadata supports,
 * everything else stays `unknown`. Never guesses fiber %, size, or brand.
 */
export class DeterministicVisionProvider implements GarmentVisionProvider {
  readonly name = 'deterministic-fallback';
  readonly version = '1.0.0';
  constructor(private meta: Record<string, string> = {}) {}

  async analyzeGarment(asset: VisualAsset): Promise<GarmentVisualParse> {
    const m = this.meta;
    const conf = (v?: string) => (v && v !== 'unknown' ? 0.9 : 0);
    return {
      ...UNKNOWN_PARSE,
      assetId: asset.id,
      garmentType: m.subCategory || m.garmentType || 'unknown',
      category: m.category || 'unknown',
      dominantColors: m.primaryColor && m.primaryColor !== 'unknown' ? [m.primaryColor] : [],
      pattern: m.pattern || 'unknown',
      silhouette: m.subCategory || 'unknown',
      materialClass: m.material ? `material-class:${m.material}` : 'unknown',
      formality: m.formality ? Number(m.formality) : 0.5,
      styleTags: [],
      confidencePerField: {
        garment_type: conf(m.subCategory || m.garmentType),
        category: conf(m.category),
        color: conf(m.primaryColor),
        pattern: conf(m.pattern),
        material: m.material ? 0.6 : 0,
      },
      provenance: 'deterministic-fallback:structured-metadata-only',
      parsedAt: new Date().toISOString(),
    };
  }

  async analyzeOutfit(asset: VisualAsset): Promise<OutfitVisualParse> {
    return {
      assetId: asset.id,
      upperBody: null, lowerBody: null, outerwear: null, shoes: null,
      accessories: [],
      dominantPalette: [],
      contrast: 'unknown',
      silhouette: 'unknown',
      layering: 1,
      formality: 0.5,
      styleDescriptors: [],
      balanceNotes: ['No visual model active; connect a provider or correct manually.'],
      confidence: 0,
      provenance: 'deterministic-fallback',
      parsedAt: new Date().toISOString(),
    };
  }

  async compareGarments(): Promise<{ score: number; notes: string[] }> {
    return { score: 0, notes: ['No visual model active; structured comparison only.'] };
  }

  async extractVisualFeatures(): Promise<Record<string, string[]>> {
    return {};
  }
}

/** Mock provider for tests/evaluation: scripted, versioned, never live. */
export class MockVisionProvider implements GarmentVisionProvider {
  readonly name = 'mock';
  readonly version = 'mock-1';
  constructor(
    private garment: Partial<GarmentVisualParse> = {},
    private outfit: Partial<OutfitVisualParse> = {},
  ) {}
  async analyzeGarment(asset: VisualAsset): Promise<GarmentVisualParse> {
    const { assetId: _g, ...garmentRest } = this.garment;
    void _g;
    return {
      ...UNKNOWN_PARSE, parsedAt: new Date().toISOString(),
      provenance: 'mock', confidencePerField: { garment_type: 0.95 },
      ...garmentRest, assetId: asset.id,
    };
  }
  async analyzeOutfit(asset: VisualAsset): Promise<OutfitVisualParse> {
    const { assetId: _o, ...outfitRest } = this.outfit;
    void _o;
    return {
      assetId: asset.id, accessories: [], dominantPalette: [], contrast: 'unknown',
      silhouette: 'unknown', layering: 1, formality: 0.5, styleDescriptors: [],
      balanceNotes: [], confidence: 0.9, provenance: 'mock',
      parsedAt: new Date().toISOString(), ...outfitRest,
    };
  }
  async compareGarments(): Promise<{ score: number; notes: string[] }> {
    return { score: 0.5, notes: ['mock comparison'] };
  }
  async extractVisualFeatures(): Promise<Record<string, string[]>> {
    return {};
  }
}

let active: GarmentVisionProvider = new DeterministicVisionProvider();
export function setVisionProvider(p: GarmentVisionProvider): void { active = p; }
export function getVisionProvider(): GarmentVisionProvider { return active; }

/** Request guards for future product-image fetching (§48). */
export const FETCH_LIMITS = {
  maxBytes: 8 * 1024 * 1024,
  timeoutMs: 15_000,
  maxRetries: 1,
  allowedProtocols: ['https:'],
} as const;

export function isSafeImageUrl(url: string): boolean {
  try {
    const u = new URL(url);
    if (!FETCH_LIMITS.allowedProtocols.includes(u.protocol as 'https:')) return false;
    if (['localhost', '127.0.0.1', '0.0.0.0', '::1'].includes(u.hostname)) return false;
    if (u.username || u.password) return false;
    return true;
  } catch {
    return false;
  }
}

export type { GarmentRegion };
