import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
const require = createRequire(import.meta.url);
const semver = require('semver');
const eas = JSON.parse(readFileSync(new URL('../../eas.json', import.meta.url), 'utf8'));
for (const name of ['react-native', 'metro', '@react-native/gradle-plugin']) {
  test(`production EAS Node satisfies installed ${name}`, () => {
    const { engines } = require(`${name}/package.json`);
    assert.ok(semver.valid(eas.build.production.node), 'Pin an exact production Node version');
    assert.ok(semver.satisfies(eas.build.production.node, engines.node), `${eas.build.production.node} does not satisfy ${name}: ${engines.node}`);
  });
}


test('Android does not override Hermes with the removed React Native SDK path', () => {
  const gradle = readFileSync(new URL('../../android/app/build.gradle', import.meta.url), 'utf8');
  assert.equal(gradle.includes('/sdks/hermesc/'), false);
  assert.ok(require.resolve('hermes-compiler/package.json'));
});


test('Android application imports native host APIs shipped by installed Expo', () => {
  const application = readFileSync(new URL('../../android/app/src/main/java/com/aiwardrobe/MainApplication.kt', import.meta.url), 'utf8');
  const root = dirname(require.resolve('expo/package.json'));
  for (const [, name] of application.matchAll(/^import expo\.modules\.(\w+)$/gm)) {
    assert.ok(['kt', 'java'].some(ext => existsSync(join(root, 'android/src/main/java/expo/modules', `${name}.${ext}`))), `Missing installed Expo native API: ${name}`);
  }
});
