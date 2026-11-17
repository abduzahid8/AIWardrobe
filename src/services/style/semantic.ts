/**
 * semantic.ts — Provider-neutral semantic + visual style abstractions
 * (Phase 3 §31–32).
 *
 * SemanticEncoder covers text/product/outfit similarity with a
 * deterministic hashing fallback so the app works fully offline. No
 * external provider is hardwired; wardrobe photos/body data never leave
 * the device through this module (it only sees caller-supplied strings
 * and structured metadata). Visual style uses existing structured
 * metadata first (category/color/pattern/material from ClothingItem).
 */

export interface SemanticEncoder {
  readonly name: string;
  encodeText(text: string): Promise<number[]> | number[];
  encodeProduct(meta: Record<string, string>): Promise<number[]> | number[];
  encodeOutfit(items: Array<Record<string, string>>): Promise<number[]> | number[];
  similarity(a: readonly number[], b: readonly number[]): number;
}

export interface VisualStyle {
  dominantColors: string[];
  contrast: 'low' | 'medium' | 'high' | 'unknown';
  silhouette: string;
  layering: number;
  formality: number;
  texture: string;
  pattern: string;
  footwearType: string;
  visualComplexity: 'minimal' | 'balanced' | 'expressive' | 'unknown';
}

/** Deterministic FNV-1a hash embedding — offline fallback, 32 dims. */
export function hashEmbedding(text: string, dims = 32): number[] {
  const vec = new Array<number>(dims).fill(0);
  const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  for (const token of tokens) {
    let h = 0x811c9dc5;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    vec[h % dims] += 1;
  }
  const norm = Math.sqrt(vec.reduce((n, v) => n + v * v, 0)) || 1;
  return vec.map((v) => Math.round((v / norm) * 1000) / 1000);
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return Math.round((dot / (Math.sqrt(na) * Math.sqrt(nb))) * 1000) / 1000;
}

export class DeterministicEncoder implements SemanticEncoder {
  readonly name = 'deterministic-fallback';
  encodeText(text: string): number[] {
    return hashEmbedding(text);
  }
  encodeProduct(meta: Record<string, string>): number[] {
    return hashEmbedding(Object.values(meta).join(' '));
  }
  encodeOutfit(items: Array<Record<string, string>>): number[] {
    return hashEmbedding(items.map((i) => Object.values(i).join(' ')).join(' | '));
  }
  similarity(a: readonly number[], b: readonly number[]): number {
    return cosineSimilarity(a, b);
  }
}

let activeEncoder: SemanticEncoder = new DeterministicEncoder();

/** Swap in a remote/provider encoder only with explicit user approval. */
export function setSemanticEncoder(enc: SemanticEncoder): void {
  activeEncoder = enc;
}

export function getSemanticEncoder(): SemanticEncoder {
  return activeEncoder;
}

/** Visual style from structured metadata — no image processing. */
export function visualStyleFromMetadata(meta: {
  primaryColor?: string;
  colors?: string[];
  pattern?: string;
  material?: string;
  subCategory?: string;
  category?: string;
  formality?: number;
  layerCount?: number;
}): VisualStyle {
  const colors = (meta.colors || (meta.primaryColor ? [meta.primaryColor] : [])).slice(0, 3);
  const neutrals = new Set(['black', 'white', 'grey', 'gray', 'navy', 'beige', 'brown', 'cream']);
  const nonNeutral = colors.filter((c) => !neutrals.has(c.toLowerCase())).length;
  const contrast: VisualStyle['contrast'] =
    colors.length < 2 ? 'unknown' : nonNeutral >= 2 ? 'high' : nonNeutral === 1 ? 'medium' : 'low';
  const patternCount = meta.pattern && meta.pattern !== 'solid' ? 1 : 0;
  const complexity: VisualStyle['visualComplexity'] =
    nonNeutral + patternCount >= 3 ? 'expressive' : nonNeutral + patternCount === 0 ? 'minimal' : 'balanced';
  return {
    dominantColors: colors,
    contrast,
    silhouette: meta.subCategory || meta.category || 'unknown',
    layering: meta.layerCount || 1,
    formality: meta.formality ?? 0.5,
    texture: meta.material || 'unknown',
    pattern: meta.pattern || 'unknown',
    footwearType: meta.category === 'shoes' ? meta.subCategory || 'shoes' : 'none',
    visualComplexity: complexity,
  };
}
