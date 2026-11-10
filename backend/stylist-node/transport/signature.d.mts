export declare const STYLIST_TRANSPORT_MAX_BODY_BYTES: number;
export declare const STYLIST6_VISION_MAX_BODY_BYTES: number;
export declare const STYLIST_TRANSPORT_MAX_SKEW_MS: number;
export interface SignedStylistTransport {
  secret: string; body: string; authorization: string; now?: number; maxBodyBytes?: number; nonce?: string;
}
export interface VerifiedStylistTransport {
  secret: string; body: string; authorization: string;
  timestamp: string | null; nonce: string | null; signature: string | null; now?: number; maxBodyBytes?: number;
}
export declare function signStylistTransport(input: SignedStylistTransport): Promise<Record<string, string>>;
export declare function verifyStylistTransport(input: VerifiedStylistTransport): Promise<boolean>;
