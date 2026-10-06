/**
 * Auth API Contract Tests
 *
 * Backend truth (backend/api/routes/auth.js mounted at "/" in backend/api/index.js:117):
 *   - Registration/login are Supabase-direct on the mobile client.
 *   - Express only exposes GET /me and GET /subscription-status,
 *     both behind `authenticateToken` (401 NO_TOKEN without a bearer token).
 *   - POST /api/auth/login and POST /api/auth/register MUST NOT exist (404).
 *
 * These tests mock `fetch` to lock that contract so a future server change
 * can't silently reintroduce password handling or drop auth guards.
 * Live server checks live in `docs/` runbooks, not in unit CI.
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3000';

const mockFetch = jest.fn<typeof fetch>();

beforeEach(() => {
  mockFetch.mockReset();
  global.fetch = mockFetch as unknown as typeof fetch;

  mockFetch.mockImplementation((async (input: any, init?: any) => {
    const urlStr = input.toString();
    const method = (init?.method ?? 'GET').toUpperCase();
    const authHeader =
      (init?.headers as Record<string, string> | undefined)?.['Authorization'] ?? '';

    // Real backend: no password endpoints — Express returns 404.
    if (urlStr.includes('/api/auth/login') || urlStr.includes('/api/auth/register')) {
      return { status: 404, json: async () => ({ error: 'Not Found' }) } as Response;
    }

    // Real backend: profile endpoints require a bearer token.
    // Mounted at "/" (see backend/api/index.js) — NOT under /api/auth.
    if (urlStr === `${API_URL}/me` || urlStr === `${API_URL}/subscription-status`) {
      if (!authHeader) {
        return {
          status: 401,
          json: async () => ({ error: 'Authentication required', code: 'NO_TOKEN' }),
        } as Response;
      }
      if (authHeader === 'Bearer invalid-token') {
        return {
          status: 401,
          json: async () => ({ error: 'Invalid or expired token', code: 'INVALID_TOKEN' }),
        } as Response;
      }
      return { status: 200, json: async () => ({ ok: true }) } as Response;
    }

    return { status: 200, json: async () => ({}) } as Response;
  }) as typeof fetch);
});

describe('Auth API Contract', () => {
  describe('password endpoints must not exist on the backend', () => {
    it('POST /api/auth/login returns 404 (auth is Supabase-direct)', async () => {
      const res = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test@test.com', password: 'wrong' }),
      });
      expect(res.status).toBe(404);
    });

    it('POST /api/auth/register returns 404 (auth is Supabase-direct)', async () => {
      const res = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'not-an-email', password: '123456' }),
      });
      expect(res.status).toBe(404);
    });
  });

  describe('GET /me', () => {
    it('returns 401 NO_TOKEN without a bearer token', async () => {
      const res = await fetch(`${API_URL}/me`);
      expect(res.status).toBe(401);
      expect(await res.json()).toMatchObject({ code: 'NO_TOKEN' });
    });

    it('returns 401 INVALID_TOKEN for a bad token', async () => {
      const res = await fetch(`${API_URL}/me`, {
        headers: { Authorization: 'Bearer invalid-token' },
      });
      expect(res.status).toBe(401);
    });
  });

  describe('GET /subscription-status', () => {
    it('returns 401 without a bearer token', async () => {
      const res = await fetch(`${API_URL}/subscription-status`);
      expect(res.status).toBe(401);
    });
  });
});
