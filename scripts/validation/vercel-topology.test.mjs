import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const read = (file) => readFileSync(new URL(file, root), 'utf8');

test('root Vercel project is the Expo web application', () => {
  const config = JSON.parse(read('vercel.json'));
  assert.equal(config.installCommand, 'npm ci');
  assert.equal(config.buildCommand, 'npx expo export --platform web');
  assert.equal(config.outputDirectory, 'dist');
  assert.equal(existsSync(new URL('package.json', root)), true);
  assert.equal(existsSync(new URL('public/package.json', root)), false);
});

test('AliceVision is an external service, not a deployable repository directory', () => {
  assert.equal(existsSync(new URL('alicevision-service/', root)), false);
  assert.match(read('docs/API_RETIREMENT_PLAN.md'), /abduzahid8\/alicevision/);
  assert.match(
    read('docs/API_RETIREMENT_PLAN.md'),
    /must not create or host an `alicevision-service\/` Vercel/
  );
});

test('Stylist 5 runtime uses Supabase Edge Functions instead of legacy AliceVision clients', () => {
  const outfitScreen = read('screens/OutfitAIScreen.tsx');
  const provider = read('src/services/aiProviderService.ts');
  assert.match(outfitScreen, /generateOutfitsFromDB/);
  assert.match(provider, /aiVisionService/);
  for (const file of [
    'src/services/visionService.ts',
    'src/services/ai/scanService.ts',
    'src/services/ai/chatService.ts',
    'src/services/ai/outfitService.ts',
  ]) {
    assert.equal(
      read(file).includes('Config.api.alicevisionUrl'),
      true,
      `${file} should remain explicitly identifiable as a legacy AliceVision client`
    );
  }
});
