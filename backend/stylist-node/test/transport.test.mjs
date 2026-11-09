import test from 'node:test';
import assert from 'node:assert/strict';
import {
  signStylistTransport,
  verifyStylistTransport,
  STYLIST_TRANSPORT_MAX_BODY_BYTES,
} from '../transport/signature.ts';

const secret = 'unit-test-only-secret-that-is-long-enough-0123456789';
const authorization = 'Bearer test-authorization';
const body = JSON.stringify({ prompt: 'Old Money', occasion: 'Work' });
const nonce = '58f7283e-c3aa-47dd-a3a7-734f12ef61c0';
const now = 1794230000000;

test('valid signed transport is accepted', async () => {
  const headers = await signStylistTransport({ secret, authorization, body, nonce, now });
  assert.equal(await verifyStylistTransport({
    secret, authorization, body, timestamp: headers['x-aiw-timestamp'],
    nonce: headers['x-aiw-nonce'], signature: headers['x-aiw-signature'], now,
  }), true);
});

test('tampering with body or bearer token is rejected', async () => {
  const headers = await signStylistTransport({ secret, authorization, body, nonce, now });
  const common = {
    secret, timestamp: headers['x-aiw-timestamp'],
    nonce: headers['x-aiw-nonce'], signature: headers['x-aiw-signature'], now,
  };
  assert.equal(await verifyStylistTransport({
    ...common, authorization, body: body + ' ',
  }), false);
  assert.equal(await verifyStylistTransport({
    ...common, authorization: 'Bearer another-user', body,
  }), false);
});

test('stale, future, invalid and absent signatures are rejected', async () => {
  const headers = await signStylistTransport({ secret, authorization, body, nonce, now });
  for (const delta of [-60_000, 60_000]) {
    assert.equal(await verifyStylistTransport({
      secret, authorization, body, timestamp: headers['x-aiw-timestamp'],
      nonce: headers['x-aiw-nonce'], signature: headers['x-aiw-signature'],
      now: now + delta,
    }), false);
  }
  assert.equal(await verifyStylistTransport({
    secret, authorization, body, timestamp: 'not-a-time',
    nonce, signature: headers['x-aiw-signature'], now,
  }), false);
  assert.equal(await verifyStylistTransport({
    secret, authorization, body, timestamp: headers['x-aiw-timestamp'],
    nonce, signature: null, now,
  }), false);
});

test('payload size and unconfigured shared secret fail closed', async () => {
  assert.rejects(() => signStylistTransport({
    secret, authorization, body: 'x'.repeat(STYLIST_TRANSPORT_MAX_BODY_BYTES + 1),
    nonce, now,
  }), /invalid_request/);
  await assert.rejects(() => signStylistTransport({
    secret: 'too-short', authorization, body, nonce, now,
  }), /unconfigured/);
});
