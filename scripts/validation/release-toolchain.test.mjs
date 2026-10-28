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

// Metro follows app imports into shared validators; EAS must ship those sources.
test('EAS archive includes shared validation code while excluding backend services and secrets',()=>{
  const ignore=require('ignore')().add(readFileSync(new URL('../../.easignore',import.meta.url),'utf8'));
  for(const name of ['budgetFx','outfitStructure','outfitComposition','outfitBudgetSelection','occasionRules']){
    const path=`backend/supabase/functions/_shared/${name}.ts`;
    assert.ok(!ignore.ignores(path),`EAS omits app dependency: ${path}`);
  }
  for(const directory of ['backend/supabase','backend/supabase/functions','backend/supabase/functions/_shared']){
    assert.ok(!ignore.ignores(directory),`EAS prunes required parent directory: ${directory}`);
  }
  for(const path of ['.env','.env.local','.env.production','backend/.env','backend/config/credentials.json','backend/supabase/migrations/private.sql','backend/supabase/functions/generate-outfits/index.ts','android/app/.cxx','android/app/build','ios/Pods','ios/build','.git'])assert.ok(ignore.ignores(path),`Unexpected archive inclusion: ${path}`);
});

test('all EAS profiles propagate peer checks into Git dependency preparation',()=>{
 for(const[name,profile]of Object.entries(eas.build))assert.equal(profile.env?.NPM_CONFIG_LEGACY_PEER_DEPS,'false',name);
});

test('EAS resolves the Appium XCUITest driver from the npm registry, not Git',()=>{
  const packageJson = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
  const lock = JSON.parse(readFileSync(new URL('../../package-lock.json', import.meta.url), 'utf8'));
  assert.equal(packageJson.devDependencies['appium-xcuitest-driver'], '12.18.1');
  assert.match(lock.packages['node_modules/appium-xcuitest-driver'].resolved, /^https:\/\/registry\.npmjs\.org\//);
});
