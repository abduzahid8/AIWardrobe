import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  analyzeUserAppearance,
  validateAppearancePhotoUrl,
  validateAppearanceAnalysisPayload,
  APPEARANCE_CACHE_VERSION,
} from './user_appearance_analyzer.mjs';

function makeValidJpegBytes(seed = 'sample-user-photo') {
  const header = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01]);
  const body = Buffer.from(seed.repeat(12), 'utf8');
  const footer = Buffer.from([0xff, 0xd9]);
  return Buffer.concat([header, body, footer]);
}

function makeValidVlmPayload(overrides = {}) {
  return {
    skinUndertone: 'warm',
    skinDepth: 'medium',
    contrastLevel: 'low',
    colorSeason: 'warm_autumn',
    flatteringColors: ['camel', 'olive', '#0A1931', 'terracotta'],
    powerNeutrals: ['espresso', 'cream', 'gray'],
    avoidColors: ['ice_blue', 'silver'],
    metalTone: 'gold',
    lightingQuality: 'good',
    confidence: 0.86,
    ...overrides,
  };
}

test('strictly opt-in: non-true consent immediately returns consent_required with zero I/O', async () => {
  let providerCalled = false;
  let fetchCalled = false;
  for (const badConsent of [false, undefined, null, 'true', 1, {}]) {
    const result = await analyzeUserAppearance({
      photoBytes: makeValidJpegBytes(),
      consent: badConsent,
      userId: 'user-123',
      providerFn: async () => { providerCalled = true; return makeValidVlmPayload(); },
      fetchFn: async () => { fetchCalled = true; throw new Error('must not fetch'); },
    });
    assert.equal(result.status, 'consent_required');
    assert.equal(result.appearance, null);
  }
  assert.equal(providerCalled, false);
  assert.equal(fetchCalled, false);
});

test('SSRF URL guard blocks http, metadata endpoints, loopback, private IPs, credentials, ports, and DNS rebinding', async () => {
  const blockedUrls = [
    'http://cdn.example.com/photo.jpg',
    'file:///etc/passwd',
    'data:image/jpeg;base64,/9j/4AAQ',
    'https://user:pass@cdn.example.com/photo.jpg',
    'https://cdn.example.com:8443/photo.jpg',
    'https://localhost/photo.jpg',
    'https://sub.localhost/photo.jpg',
    'https://127.0.0.1/photo.jpg',
    'https://169.254.169.254/latest/meta-data/',
    'https://metadata.google.internal/computeMetadata/v1/',
    'https://10.0.0.4/internal.jpg',
    'https://192.168.1.10/photo.jpg',
    'https://172.16.5.2/photo.jpg',
    'https://[::1]/photo.jpg',
    'https://2130706433/photo.jpg',
    'https://0x7f000001/photo.jpg',
    'https://internal.local/photo.jpg',
  ];

  for (const url of blockedUrls) {
    assert.equal(validateAppearancePhotoUrl(url).ok, false, `expected URL blocked: ${url}`);
    const res = await analyzeUserAppearance({
      photoUrl: url,
      consent: true,
      providerFn: async () => makeValidVlmPayload(),
    });
    assert.equal(res.status, 'blocked_url', `expected blocked_url for ${url}`);
    assert.equal(res.appearance, null);
  }

  // DNS rebinding check: public-looking domain resolving to 127.0.0.1 or 10.0.0.1
  for (const privateIp of ['127.0.0.1', '10.1.2.3', '169.254.169.254', '::1', 'fe80::1', '::ffff:127.0.0.1']) {
    const res = await analyzeUserAppearance({
      photoUrl: 'https://photos.example.com/user-selfie.jpg',
      consent: true,
      dnsLookupFn: async () => [{ address: privateIp }],
      fetchFn: async () => { throw new Error('must not fetch private DNS IP'); },
      providerFn: async () => makeValidVlmPayload(),
    });
    assert.equal(res.status, 'blocked_url');
    assert.equal(res.reason, 'dns_resolved_private_ip');
  }

  // Host allowlist enforcement
  assert.equal(
    validateAppearancePhotoUrl('https://evil.example.com/a.jpg', { allowedHosts: ['cdn.aiwardrobe.app'] }).ok,
    false,
  );
  assert.equal(
    validateAppearancePhotoUrl('https://cdn.aiwardrobe.app/a.jpg', { allowedHosts: ['cdn.aiwardrobe.app'] }).ok,
    true,
  );
});

test('SHA-256 content-hash cache hits on identical bytes and sends ZERO PII to provider', async () => {
  const bytes = makeValidJpegBytes('unique-portrait-bytes');
  const expectedSha = createHash('sha256').update(bytes).digest('hex');
  const cache = new Map();
  let calls = 0;
  let capturedProviderRequest = null;

  const providerFn = async (req) => {
    calls += 1;
    capturedProviderRequest = req;
    return makeValidVlmPayload();
  };

  // First call: cache miss -> calls provider
  const first = await analyzeUserAppearance({
    photoUrl: 'https://cdn.aiwardrobe.app/private/alice@example.com/portrait.jpg?sig=SECRET_TOKEN',
    consent: true,
    userId: 'user-uuid-9999-secret',
    cache,
    dnsLookupFn: async () => [{ address: '93.184.216.34' }],
    fetchFn: async (_url, opts) => {
      assert.equal(opts.redirect, 'error');
      return { ok: true, status: 200, arrayBuffer: async () => bytes };
    },
    providerFn,
  });

  assert.equal(first.status, 'ok');
  assert.equal(first.cached, false);
  assert.equal(first.photoSha256, expectedSha);
  assert.equal(first.cacheKey, `${APPEARANCE_CACHE_VERSION}:${expectedSha}`);
  assert.equal(calls, 1);

  // Verify canonicalization on VLM output (#0A1931 -> navy, gray -> grey)
  assert.ok(first.appearance.flatteringColors.includes('navy'));
  assert.ok(first.appearance.powerNeutrals.includes('grey'));
  assert.equal(first.appearance.provenance, 'opt_in_photo_analysis');

  // Verify ZERO PII was sent to providerFn
  const serializedReq = JSON.stringify(capturedProviderRequest);
  assert.doesNotMatch(serializedReq, /user-uuid-9999-secret|alice@example\.com|SECRET_TOKEN|cdn\.aiwardrobe\.app/);
  assert.equal(capturedProviderRequest.mimeType, 'image/jpeg');

  // Second call with identical bytes -> cache hit, provider NOT called again
  const second = await analyzeUserAppearance({
    photoBytes: bytes,
    consent: true,
    userId: 'different-user-id',
    cache,
    providerFn,
  });
  assert.equal(second.status, 'ok');
  assert.equal(second.cached, true);
  assert.equal(second.photoSha256, expectedSha);
  assert.equal(calls, 1);
});

test('strict JSON schema validation strips forbidden protected traits/body measurements and degrades low confidence to unknown', async () => {
  const bytes = makeValidJpegBytes('protected-strip-test');
  const cache = new Map();

  // 1. Hallucinated protected attributes & invented body measurements must NEVER appear in output
  const withForbiddenExtras = await analyzeUserAppearance({
    photoBytes: bytes,
    consent: true,
    cache,
    providerFn: async () => ({
      ...makeValidVlmPayload(),
      ethnicity: 'should_never_exist',
      race: 'should_never_exist',
      age: 29,
      weightKg: 74,
      heightCm: 179,
      bodyType: 'athletic',
      measurements: { waist: 81, chest: 100 },
    }),
  });
  assert.equal(withForbiddenExtras.status, 'ok');
  assert.equal('ethnicity' in withForbiddenExtras.appearance, false);
  assert.equal('race' in withForbiddenExtras.appearance, false);
  assert.equal('age' in withForbiddenExtras.appearance, false);
  assert.equal('weightKg' in withForbiddenExtras.appearance, false);
  assert.equal('heightCm' in withForbiddenExtras.appearance, false);
  assert.equal('bodyType' in withForbiddenExtras.appearance, false);
  assert.equal('measurements' in withForbiddenExtras.appearance, false);

  // 2. Low confidence degrades all attributes to 'unknown' and is NOT cached
  const lowConfCache = new Map();
  const lowConf = await analyzeUserAppearance({
    photoBytes: makeValidJpegBytes('dim-photo'),
    consent: true,
    cache: lowConfCache,
    providerFn: async () => makeValidVlmPayload({ confidence: 0.42 }),
  });
  assert.equal(lowConf.status, 'low_confidence');
  assert.equal(lowConf.appearance.lowConfidenceFallback, true);
  assert.equal(lowConf.appearance.skinUndertone, 'unknown');
  assert.equal(lowConf.appearance.skinDepth, 'unknown');
  assert.equal(lowConf.appearance.contrastLevel, 'unknown');
  assert.equal(lowConf.appearance.colorSeason, 'unknown');
  assert.equal(lowConf.appearance.metalTone, 'unknown');
  assert.deepEqual(lowConf.appearance.flatteringColors, []);
  assert.equal(lowConfCache.size, 0);

  // 3. Poor lighting degrades to 'unknown' even if model over-reports confidence
  const poorLight = validateAppearanceAnalysisPayload(
    makeValidVlmPayload({ lightingQuality: 'poor', confidence: 0.91 }),
  );
  assert.equal(poorLight.ok, true);
  assert.equal(poorLight.lowConfidence, true);
  assert.equal(poorLight.appearance.skinUndertone, 'unknown');

  // 4. Invalid schema / malformed JSON rejected cleanly
  assert.equal(validateAppearanceAnalysisPayload('not-json').ok, false);
  assert.equal(validateAppearanceAnalysisPayload({ skinUndertone: 'martian', confidence: 0.9 }).ok, false);
  assert.equal(validateAppearanceAnalysisPayload(makeValidVlmPayload({ confidence: 1.5 })).ok, false);

  // 5. Non-image bytes rejected before provider call
  const badBytes = await analyzeUserAppearance({
    photoBytes: Buffer.from('<html><body>Not a real JPEG photo at all — plain HTML response body</body></html>'),
    consent: true,
    providerFn: async () => makeValidVlmPayload(),
  });
  assert.equal(badBytes.status, 'invalid_photo');
  assert.equal(badBytes.reason, 'unsupported_image_format');

  // 6. Contradictory canonical colors in avoidColors vs flatteringColors/powerNeutrals are stripped cleanly
  const deduped = validateAppearanceAnalysisPayload(makeValidVlmPayload({
    flatteringColors: ['emerald green', 'royal blue'],
    powerNeutrals: ['navy', 'white'],
    avoidColors: ['olive green', 'orange', 'navy'],
  }));
  assert.equal(deduped.ok, true);
  assert.ok(deduped.appearance.flatteringColors.includes('green'));
  assert.ok(!deduped.appearance.avoidColors.includes('green'), 'avoidColors must not contradict flatteringColors');
  assert.ok(!deduped.appearance.avoidColors.includes('navy'), 'avoidColors must not contradict powerNeutrals');
  assert.ok(deduped.appearance.avoidColors.includes('orange'));
});
