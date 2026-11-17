/**
 * wardrobeViews.ts — Shared lightweight wardrobe views (Phase 3).
 *
 * Minimal structural views so style services stay decoupled from the full
 * ClothingItem domain and from each other (no import cycles). Adapters from
 * real domain types live here; Phase 1/2 contracts are untouched.
 */

import type { ClothingItem } from '../../types/domain';

export interface WardrobeItemView {
  id: string;
  userId: string;
  category: string;
  subCategory: string;
  primaryColor: string;
  brand?: string;
  occasions: string[];
  seasons: string[];
  wearCount: number;
  lastWornAt: string | null;
  isFavorite: boolean;
  acquiredAt?: string | null;
  condition?: string | null;
}

export function toWardrobeItemView(item: ClothingItem): WardrobeItemView {
  return {
    id: item.id,
    userId: item.userId,
    category: item.category,
    subCategory: item.subCategory || 'other',
    primaryColor: (item.primaryColor || 'unknown').toLowerCase(),
    brand: item.brand,
    occasions: [...(item.occasions || [])],
    seasons: [...(item.seasons || [])],
    wearCount: item.wearCount || 0,
    lastWornAt: item.lastWornAt,
    isFavorite: !!item.isFavorite,
  };
}

export function roleKeyOfView(item: Pick<WardrobeItemView, 'category' | 'subCategory' | 'primaryColor'>): string {
  return `${item.category}:${item.subCategory.toLowerCase()}:${item.primaryColor.toLowerCase()}`;
}

export function daysSince(dateIso: string | null, now = Date.now()): number | null {
  if (!dateIso) return null;
  const t = Date.parse(dateIso);
  if (!Number.isFinite(t)) return null;
  return Math.max(0, Math.round((now - t) / 86_400_000));
}
