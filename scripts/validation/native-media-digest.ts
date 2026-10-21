import * as Crypto from 'expo-crypto';

/** Android's installed native digest expects a TypedArray, not a bare ArrayBuffer. */
export async function hashAuditBytes(bytes: ArrayBuffer): Promise<string> {
  const digest = await Crypto.digest(Crypto.CryptoDigestAlgorithm.SHA256, new Uint8Array(bytes));
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
}
