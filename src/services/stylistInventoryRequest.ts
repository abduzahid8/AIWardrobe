import { outfitGenerationFailure } from './outfitGenerationFailure';

/** Rotation is a preference. Exhausting unseen inventory must not make real stock unavailable. */
export async function requestWithInventoryRotation<T>(
  invoke: (excludeItemIds: string[]) => Promise<{ data: T; error: unknown }>,
  excludeItemIds: string[],
  signal?: AbortSignal,
): Promise<{ data: T; error: unknown }> {
  const first = await invoke(excludeItemIds);
  if (signal?.aborted || !excludeItemIds.length) return first;
  const data = first.data as { success?: boolean } | null;
  if (!first.error && data?.success !== false) return first;
  const failure = await outfitGenerationFailure(first.data, first.error);
  if (signal?.aborted || failure.translationKey !== 'outfitMaker.noMatchingClothing') return first;
  // One retry, preserving every source, style, activity and budget constraint.
  return invoke([]);
}
