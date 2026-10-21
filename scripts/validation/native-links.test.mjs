import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { XMLParser } from 'fast-xml-parser';
const app = JSON.parse(readFileSync(new URL('../../app.json', import.meta.url), 'utf8'));
const manifest = new XMLParser({ ignoreAttributes: false }).parse(readFileSync(process.env.NATIVE_MANIFEST_PATH || new URL('../../android/app/src/main/AndroidManifest.xml', import.meta.url), 'utf8'));
const list = value => value === undefined ? [] : Array.isArray(value) ? value : [value];
const iosPlist = JSON.parse(execFileSync('python3', ['-c', 'import plistlib,json,sys; print(json.dumps(plistlib.load(open(sys.argv[1], "rb"))))',
  process.env.NATIVE_IOS_PLIST_PATH || fileURLToPath(new URL('../../ios/AIWardrobe/Info.plist', import.meta.url))], { encoding: 'utf8' }));
const activity = list(manifest.manifest.application.activity).find(value => ['.MainActivity', `${app.expo.android.package}.MainActivity`].includes(value['@_android:name']));
for (const scheme of list(app.expo.scheme)) {
  test(`iOS registers configured ${scheme} links`, () => {
    assert.ok(list(iosPlist.CFBundleURLTypes).some(value => list(value.CFBundleURLSchemes).includes(scheme)),
      `Native Info.plist does not register ${scheme}`);
  });
  test(`Android receives configured ${scheme} links`, () => {
    const filter = list(activity['intent-filter']).find(value => list(value.data).some(data => data['@_android:scheme'] === scheme));
    assert.ok(filter, `MainActivity has no intent filter for ${scheme}`);
    assert.ok(list(filter.action).some(value => value['@_android:name'] === 'android.intent.action.VIEW'));
    for (const category of ['DEFAULT', 'BROWSABLE']) {
      assert.ok(list(filter.category).some(value => value['@_android:name'] === `android.intent.category.${category}`));
    }
    assert.equal(activity['@_android:exported'], 'true');
    // apkanalyzer prints the compiled enum value (ActivityInfo.LAUNCH_SINGLE_TASK = 2).
    assert.ok(['singleTask', '2'].includes(activity['@_android:launchMode']));
  });
}
