/** Neutral retrieval wire contract shared by mobile discovery and Edge.
 * These candidates are evidence, never authority to bypass outfit or FX gates.
 */
export interface CatalogCandidateRpcRequest {
  p_slot: string;
  p_source: string;
  p_search: string;
  p_seed: string;
  p_limit: number;
  p_offset: number;
  p_currency: string | null;
  p_max_price: number | null;
}
export interface CatalogCandidateRetrievalDiagnostics {
  seed: string;
  pages: number;
  degraded: boolean;
  candidateCount: number;
}
