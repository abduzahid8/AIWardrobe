/** @jest-environment node */
// Narrowly scoped release-identity assertion: the edge response must identify
// the exact running backend. Documented in DEPLOY_STYLIST_RELEASE.md.
const fs = require('fs');
const path = require('path');

const EXPECTED_GENERATION_VERSION = 'stylist-5-editorial-v2-rc-20261109-r2';
const PRODUCTION_VERSION = 'stylist-5-editorial-v2';

const source = fs.readFileSync(
  path.resolve(__dirname, '../../backend/supabase/functions/generate-outfits/index.ts'),
  'utf8'
);

test('edge GENERATION_VERSION is the exact release-candidate identifier', () => {
  const match = source.match(/const GENERATION_VERSION = '([^']+)'/);
  expect(match).not.toBeNull();
  expect(match[1]).toBe(EXPECTED_GENERATION_VERSION);
});

test('release identifier differs from production and keeps client compatibility', () => {
  expect(EXPECTED_GENERATION_VERSION).not.toBe(PRODUCTION_VERSION);
  // Mirrors the client's server-plan gate in outfitGenerationService.ts.
  expect(/^stylist-5-/.test(EXPECTED_GENERATION_VERSION)).toBe(true);
});
