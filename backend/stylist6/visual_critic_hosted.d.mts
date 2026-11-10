export const HOSTED_VISUAL_CRITIC_VERSION: string;
export const NVIDIA_VISION_MODEL: string;

export interface HostedVisualVerdict {
  candidateId: string;
  visualCohesion: number;
  styleFit: number;
  occasionFit: number;
  formalityFit: number;
  confidence: number;
  evidence: string[];
  reject?: boolean;
  criticVersion: string;
  latencyMs: number;
  estimatedCostUsd: number | null;
}

export interface HostedVisualCandidate {
  id: string;
  briefId: string;
  concept: string;
  referenceId?: string | null;
  products: Array<{ id: string; imageUrl: string; name?: string }>;
}

export interface HostedVisualRequest {
  style?: string;
  occasion?: string;
}

export function createHostedVisualCritic(options?: {
  apiKey?: string;
  provider?: string;
  model?: string;
  endpoint?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  maxImageBytes?: number;
  estimatedCostUsdPerCall?: number;
  resolveReference?: ((referenceId: string) => Promise<{
    base64: string;
    mimeType?: string;
    sha256?: string;
  } | null>) | null;
  onEvent?: (event: { candidateId: string; outcome: string; reason?: string; latencyMs?: number }) => void;
}): (payload: HostedVisualCandidate[], request?: HostedVisualRequest) => Promise<HostedVisualVerdict[]>;
