/**
 * One authenticated transport between the existing Supabase endpoint and
 * the private Stylist Node compute function. Not a user-auth substitute:
 * the original generation handler still verifies the user's JWT and quota.
 *
 * The signature covers both the bearer token and the exact UTF-8 body, so
 * changing either during transit invalidates the request. No credentials
 * or garment/user data are ever written to the logs.
 */
export const STYLIST_TRANSPORT_MAX_BODY_BYTES = 1024 * 1024;
export const STYLIST_TRANSPORT_MAX_SKEW_MS = 45_000;

const encoder = new TextEncoder();
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SIGNATURE_RE = /^[0-9a-f]{64}$/i;

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}

function fromHex(value: string): Uint8Array {
  const bytes = new Uint8Array(value.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(value.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}

async function keyFor(secret: string): Promise<CryptoKey> {
  if (typeof secret !== 'string' || secret.length < 32) throw new Error('stylist_transport_secret_unconfigured');
  return crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
}

async function signedMessage(
  timestamp: string,
  nonce: string,
  authorization: string,
  body: string
): Promise<Uint8Array> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(body)));
  return encoder.encode([timestamp, nonce, authorization, hex(digest)].join('\n'));
}

export async function signStylistTransport(input: {
  secret: string;
  body: string;
  authorization: string;
  now?: number;
  nonce?: string;
}): Promise<Record<string, string>> {
  const timestamp = String(input.now ?? Date.now());
  const nonce = input.nonce ?? crypto.randomUUID();
  if (!UUID_RE.test(nonce) || encoder.encode(input.body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    throw new Error('stylist_transport_invalid_request');
  }
  const signature = new Uint8Array(await crypto.subtle.sign(
    'HMAC', await keyFor(input.secret),
    await signedMessage(timestamp, nonce, input.authorization, input.body)
  ));
  return {
    'x-aiw-timestamp': timestamp,
    'x-aiw-nonce': nonce,
    'x-aiw-signature': hex(signature),
  };
}

export async function verifyStylistTransport(input: {
  secret: string;
  body: string;
  authorization: string;
  timestamp: string | null;
  nonce: string | null;
  signature: string | null;
  now?: number;
}): Promise<boolean> {
  const timestamp = input.timestamp ?? '';
  const nonce = input.nonce ?? '';
  const signature = input.signature ?? '';
  const millis = Number(timestamp);
  if (!/^\d{13}$/.test(timestamp) || !Number.isSafeInteger(millis) ||
      Math.abs((input.now ?? Date.now()) - millis) > STYLIST_TRANSPORT_MAX_SKEW_MS ||
      !UUID_RE.test(nonce) || !SIGNATURE_RE.test(signature) ||
      !/^Bearer\s+\S+$/i.test(input.authorization) ||
      encoder.encode(input.body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) return false;
  try {
    return await crypto.subtle.verify('HMAC', await keyFor(input.secret),
      fromHex(signature),
      await signedMessage(timestamp, nonce, input.authorization, input.body)
    );
  } catch {
    return false;
  }
}
