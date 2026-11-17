/** Zero denotes unknown under the existing numeric vision wire contract. */
export function normalizeVisionConfidence(value: unknown): number {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : 0;
}
export type InferencePolicy = 'standard' | 'verify' | 'deep';
/** Explicit deep requests remain bounded; standard work always gets one pass. */
export function inferenceCallLimit(policy: InferencePolicy, requested = 1, ambiguous = false): number {
    if (policy === 'standard') return 1;
    if (policy === 'verify') return ambiguous ? 2 : 1;
    return Number.isFinite(requested) ? Math.max(1, Math.min(3, Math.floor(requested))) : 1;
}
