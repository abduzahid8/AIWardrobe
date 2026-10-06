/**
 * Shared TanStack QueryClient singleton.
 *
 * Lives in its own module (instead of inside App.tsx) so non-component code —
 * e.g. `clearAllPersistedUserData()` on logout / account deletion — can
 * invalidate the cache. Without this, React Query serves user A's query
 * results to user B for up to `staleTime` (5 min) after an account switch.
 */

import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 15 * 60 * 1000, // drop unused caches — caps JS memory on long sessions
      retry: 1, // 2 retries doubled network storms on slow connections (freeze)
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always',
      structuralSharing: true, // unchanged data keeps referential equality → fewer re-renders
    },
  },
});

/**
 * Drop every cached query. Call on logout, account deletion, and before
 * initializing state for a newly signed-in user.
 */
export function clearQueryCache(): void {
  try {
    queryClient.clear();
  } catch {
    // Best-effort — never block auth flows on cache clearing.
  }
}
