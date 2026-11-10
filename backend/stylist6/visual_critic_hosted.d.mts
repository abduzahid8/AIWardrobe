import type { Buffer } from 'node:buffer';

export const HOSTED_VISUAL_CRITIC_VERSION: string;
export const NVIDIA_VISION_MODEL: string;
export const MAX_VISUAL_CANDIDATES: number;
export const MAX_VISUAL_IMAGE_BYTES: number;
export const VISUAL_TIMEOUT_MS: number;
export const VISUAL_PROVIDER_TIMEOUT_MS: number;
export const VISUAL_CONCURRENCY: number;

export interface VisualImage {
  bytes: Buffer;
  sha256: string;
}

export interface ReferenceImage extends VisualImage {
  components?: { referenceSha256: string; slotSha256: Partial<Record<HostedVisualProduct['slot'], string>> };
}

export interface HostedVisualVerdict {
  candidateId: string;
  slotObservations?: Partial<Record<HostedVisualProduct['slot'], string>>;
  verifiedSharedSlots?: HostedVisualProduct['slot'][];
  verifiedReferenceOnlySlots?: HostedVisualProduct['slot'][];
  relationships?: { kind: 'outfit' | 'reference'; slots: HostedVisualProduct['slot'][]; observation: string }[];
  visualCohesion: number;
  referenceFidelity: number;
  styleFit: number;
  occasionFit: number;
  formalityFit: number;
  confidence: number;
  evidence: string[];
  reject: boolean;
  criticVersion: string;
  latencyMs: number;
  estimatedCostUsd: null;
}

export interface HostedVisualProduct {
  id: string;
  slot: 'top' | 'outerwear' | 'bottom' | 'shoes';
  imageUrl: string;
}

export interface HostedVisualCandidate {
  id: string;
  referenceId?: string | null;
  products: HostedVisualProduct[];
}

export interface HostedVisualRequest {
  style?: string;
  occasion?: string;
  brief?: string;
}

export function validVisualVerdict(row: unknown, candidateId: string): row is HostedVisualVerdict;
export function verifiedVisualImage(asset: VisualImage | null): { mimeType: string; data: string; sha256: string };

export function createHostedVisualCritic(options: {
  apiKey: string;
  provider?: 'nvidia' | 'gemini';
  model?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  providerTimeoutMs?: number;
  signal?: AbortSignal;
  resolveReference: (referenceId: string, signal: AbortSignal) => Promise<ReferenceImage | null> | ReferenceImage | null;
  resolveProduct: (product: HostedVisualProduct, signal: AbortSignal) => Promise<VisualImage | null> | VisualImage | null;
  onEvent?: (event: {
    candidateId: string;
    outcome: string;
    dispatched: boolean;
    reason?: string;
    latencyMs?: number;
    queueMs: number;
    mediaMs?: number;
    providerMs?: number;
    headersMs?: number;
    parseMs?: number;
    validationMs?: number;
    httpStatus?: number;
    finishReason?: string;
    usage?: Partial<Record<'prompt_tokens' | 'completion_tokens' | 'total_tokens', number>>;
  }) => void;
}): (payload: HostedVisualCandidate[], request?: HostedVisualRequest) => Promise<HostedVisualVerdict[]>;
