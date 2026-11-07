import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const read = (file) => readFileSync(file, 'utf8'),
  json = (file) => JSON.parse(read(file));
const app = json('app.json').expo,
  pkg = json('package.json');
assert.equal(app.version, pkg.version);
assert.match(
  read('android/app/build.gradle'),
  new RegExp(`versionCode ${app.android.versionCode}\\b`)
);
assert.match(
  read('ios/AIWardrobe.xcodeproj/project.pbxproj'),
  new RegExp(`CURRENT_PROJECT_VERSION = ${app.ios.buildNumber};`)
);
const en = json('i18n/locales/en.json');
const flatten = (object, prefix = '') =>
  Object.entries(object).flatMap(([key, value]) =>
    typeof value === 'object' && value !== null
      ? flatten(value, `${prefix}${key}.`)
      : [`${prefix}${key}`]
  );
for (const lang of ['ru', 'uz'])
  assert.deepEqual(
    flatten(json(`i18n/locales/${lang}.json`).beta).sort(),
    flatten(en.beta).sort(),
    `beta translation key parity: ${lang}`
  );
const flags = read('src/services/platform/experimentation.ts');
for (const flag of [
  'autonomous_stylist',
  'proactive_recommendations',
  'price_intelligence',
  'visual_parser',
  'new_ranking',
])
  assert.match(flags, new RegExp(`${flag}: false`));
const cfg = json('docs/launch/activation-config.json');
assert.equal(new Set(cfg.pendingMigrations).size, cfg.pendingMigrations.length);
assert.deepEqual(cfg.pendingMigrations, [...cfg.pendingMigrations].sort());
for (const migration of cfg.pendingMigrations)
  assert.ok(read(`backend/supabase/migrations/${migration}`).length);
console.log(
  'PASS version/native-build consistency, beta locale key parity, conservative platform defaults, explicit pending migration order'
);
console.log(
  'MANUAL REQUIRED full-app translation review, remote flag wiring/drills, screen-reader QA'
);

const stopTemplate = read('backend/supabase/functions/.env.example');
assert.match(stopTemplate, /^AIWARDROBE_PAID_WORK_DISABLED=true$/m);
assert.match(stopTemplate, /^AIWARDROBE_DISABLED_PAID_FEATURES=try-on$/m);
assert.doesNotMatch(stopTemplate, /^EXPO_PUBLIC_/m);
assert.equal(cfg.serverPaidWorkControls.requiresExplicitApproval, true);
assert.equal(cfg.serverPaidWorkControls.preparationPosture.AIWARDROBE_PAID_WORK_DISABLED, 'true');
console.log('PASS server-only paused preparation template and explicit activation approval record');
