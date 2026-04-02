/**
 * src/lib/api.ts — Typed API client for the AIWardrobe backend
 *
 * All requests automatically attach the Supabase session token.
 * Retries once on 401 (token refresh), then throws ApiError.
 * Never use raw fetch in screens — always go through this client.
 */

import { supabase } from '../../lib/supabase';

// ── Config ─────────────────────────────────────────────────────────────────

const BASE_URL = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3001').replace(/\/$/, '');
const API_V1 = `${BASE_URL}/api/v1`;

// ── Error type ─────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly code?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// ── Token helper ───────────────────────────────────────────────────────────

async function getToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

// ── Core request ───────────────────────────────────────────────────────────

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  retried = false
): Promise<T> {
  const token = await getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_V1}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (res.status === 401 && !retried) {
    // Attempt token refresh once
    await supabase.auth.refreshSession();
    return request<T>(method, path, body, true);
  }

  if (res.status === 204) return undefined as unknown as T;

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new ApiError(
      res.status,
      (data as { error?: string }).error ?? `HTTP ${res.status}`,
      (data as { code?: string }).code
    );
  }

  return data as T;
}

const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  patch: <T>(path: string, body: unknown) => request<T>('PATCH', path, body),
  delete: <T = void>(path: string) => request<T>('DELETE', path),
};

// ── Domain types (mirroring backend serializers) ───────────────────────────

export interface ApiClothingItem {
  id: string;
  userId: string;
  closetId: string | null;
  imageUrl: string;
  thumbnailUrl: string | null;
  cutoutUrl: string | null;
  category: string;
  subCategory: string;
  primaryColor: string;
  colorHex: string;
  pattern: string;
  material: string;
  layer: string | null;
  brand: string | null;
  name: string | null;
  seasons: string[];
  occasions: string[];
  wearCount: number;
  lastWornAt: string | null;
  isFavorite: boolean;
  detectionConfidence: number | null;
  aiDescription: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ApiOutfitItem {
  id: string;
  imageUrl: string;
  thumbnailUrl: string | null;
  category: string;
  subCategory: string;
  primaryColor: string;
  name: string | null;
}

export interface ApiOutfit {
  id: string;
  userId: string;
  occasion: string;
  generatedBy: 'ai' | 'user';
  previewImageUrl: string | null;
  reasoning: string | null;
  colorHarmony: string | null;
  style: string | null;
  formalityTier: number | null;
  saved: boolean;
  wornCount: number;
  lastWornAt: string | null;
  rating: number | null;
  itemIds: string[];
  items: ApiOutfitItem[];
  createdAt: string;
  updatedAt: string;
}

export interface ApiWearLog {
  id: string;
  userId: string;
  outfitId: string | null;
  date: string;
  occasion: string | null;
  weatherTemp: number | null;
  weatherCondition: string | null;
  notes: string | null;
  itemIds: string[];
  createdAt: string;
}

export interface ApiUser {
  id: string;
  email: string;
  username: string | null;
  gender: string | null;
  profileImage: string | null;
  preferredStyles: string[];
  preferredColors: string[];
  avoidColors: string[];
  bodyType: string | null;
  stylePersonality: string | null;
  tier: string;
  tierExpiresAt: string | null;
  onboardingComplete: boolean;
  streakDays: number;
  lastActiveAt: string;
  createdAt: string;
}

export interface ApiCloset {
  id: string;
  name: string;
  description: string | null;
  isDefault: boolean;
  coverImage: string | null;
  itemCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface ApiAnalytics {
  totalItems: number;
  totalOutfits: number;
  totalWearLogs: number;
  streakDays: number;
  closetUtilization30d: number;
  closetUtilization7d: number;
  categoryBreakdown: Record<string, number>;
  colorFrequency: Array<{ color: string; count: number }>;
  mostWornItems: Array<{ id: string; name: string; wearCount: number; imageUrl: string }>;
  leastWornItems: Array<{ id: string; name: string; wearCount: number; imageUrl: string }>;
  averageWearsPerItem: number;
  computedAt: string;
}

export interface ApiChatResponse {
  response: string;
}

export interface ApiClothingDetection {
  category: string;
  subCategory: string;
  primaryColor: string;
  colorHex: string;
  pattern: string;
  material: string;
  confidence: number;
  aiDescription: string;
}

export interface ApiGeneratedOutfits {
  outfits: Array<{
    itemIds: string[];
    reasoning: string;
    items: Array<{ id: string; imageUrl?: string; category: string; name?: string | null }>;
  }>;
}

export interface ApiUploadEnqueueResponse {
  id: string;
  tempId: string;
  status: string;
  enqueuedAt: string;
}

export interface ApiUploadStatusItem {
  id: string;
  tempId: string;
  thumbnailUrl: string | null;
  label: string | null;
  status: 'PENDING' | 'PROCESSING' | 'SUCCEEDED' | 'FAILED';
  retryCount: number;
  errorMessage: string | null;
  resultItemId: string | null;
  enqueuedAt: string;
  processedAt: string | null;
}

// ── Wardrobe endpoints ─────────────────────────────────────────────────────

export const wardrobeApi = {
  list: () =>
    api.get<ApiClothingItem[]>('/wardrobe'),

  add: (item: Partial<ApiClothingItem>) =>
    api.post<ApiClothingItem>('/wardrobe', item),

  update: (id: string, updates: Partial<ApiClothingItem>) =>
    api.patch<ApiClothingItem>(`/wardrobe/${id}`, updates),

  remove: (id: string) =>
    api.delete(`/wardrobe/${id}`),

  toggleFavorite: (id: string) =>
    api.post<{ id: string; isFavorite: boolean }>(`/wardrobe/${id}/favorite`),
};

// ── Outfit endpoints ───────────────────────────────────────────────────────

export const outfitsApi = {
  list: (all = false) =>
    api.get<ApiOutfit[]>(`/outfits${all ? '?all=true' : ''}`),

  create: (payload: {
    itemIds: string[];
    occasion: string;
    generatedBy?: 'AI' | 'USER';
    previewImageUrl?: string;
    reasoning?: string;
    style?: string;
  }) => api.post<ApiOutfit>('/outfits', payload),

  update: (id: string, updates: Partial<ApiOutfit>) =>
    api.patch<ApiOutfit>(`/outfits/${id}`, updates),

  remove: (id: string) =>
    api.delete(`/outfits/${id}`),

  toggleSave: (id: string) =>
    api.post<{ id: string; saved: boolean }>(`/outfits/${id}/save`),

  rate: (id: string, rating: 1 | 2 | 3 | 4 | 5) =>
    api.post<{ id: string; rating: number }>(`/outfits/${id}/rate`, { rating }),
};

// ── Wear log endpoints ─────────────────────────────────────────────────────

export const wearLogApi = {
  list: (since?: string) =>
    api.get<ApiWearLog[]>(`/wear-logs${since ? `?since=${since}` : ''}`),

  log: (payload: {
    itemIds: string[];
    outfitId?: string;
    date: string;
    occasion?: string;
    weatherTemp?: number;
    weatherCondition?: string;
    notes?: string;
  }) => api.post<ApiWearLog>('/wear-logs', payload),

  remove: (id: string) =>
    api.delete(`/wear-logs/${id}`),
};

// ── AI endpoints ───────────────────────────────────────────────────────────

export const aiApi = {
  chat: (payload: {
    message: string;
    history?: Array<{ role: 'user' | 'model'; content: string }>;
    weatherCity?: string;
  }) => api.post<ApiChatResponse>('/ai/chat', payload),

  analyzeClothing: (imageBase64: string) =>
    api.post<ApiClothingDetection>('/ai/analyze-clothing', { imageBase64 }),

  generateOutfits: (payload: {
    occasion?: string;
    weatherCity?: string;
    count?: number;
  }) => api.post<ApiGeneratedOutfits>('/ai/generate-outfits', payload),

  getDailySuggestion: () =>
    api.get<ApiOutfit>('/ai/daily-suggestion'),
};

// ── Analytics endpoints ────────────────────────────────────────────────────

export const analyticsApi = {
  get: () => api.get<ApiAnalytics>('/analytics'),
};

// ── Upload endpoints ───────────────────────────────────────────────────────

export const uploadApi = {
  enqueue: (payload: {
    tempId: string;
    imageStoragePath: string;
    thumbnailUrl?: string;
    label?: string;
  }) => api.post<ApiUploadEnqueueResponse>('/upload/enqueue', payload),

  status: () =>
    api.get<ApiUploadStatusItem[]>('/upload/status'),

  cancel: (tempId: string) =>
    api.delete(`/upload/${tempId}`),
};

// ── User endpoints ─────────────────────────────────────────────────────────

export const userApi = {
  getMe: () =>
    api.get<ApiUser>('/user/me'),

  updateMe: (updates: Partial<ApiUser> & { expoPushToken?: string }) =>
    api.patch<ApiUser>('/user/me', updates),

  deleteMe: () =>
    api.delete('/user/me'),

  getClosets: () =>
    api.get<ApiCloset[]>('/user/me/closets'),

  createCloset: (payload: { name: string; description?: string; coverImage?: string }) =>
    api.post<ApiCloset>('/user/me/closets', payload),
};

export default api;
