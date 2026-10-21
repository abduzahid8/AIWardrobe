import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const core = dirname(require.resolve('@react-navigation/core/package.json'));
const { getStateFromPath } = await import(pathToFileURL(join(core, 'lib/module/getStateFromPath.js')));
const { getPathFromState } = await import(pathToFileURL(join(core, 'lib/module/getPathFromState.js')));
// Read the real app config rather than a second route map that can drift.
const app = ts.createSourceFile('App.tsx', readFileSync(new URL('../../App.tsx', import.meta.url), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let initializer;
function visit(node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(app) === 'linking') initializer = node.initializer.getText(app);
  ts.forEachChild(node, visit);
}
visit(app);
assert.ok(initializer, 'App linking configuration exists');
const { config } = Function('Linking', `return (${initializer})`)({ createURL: () => 'com.aiwardrobe://' });

for (const [name, path] of Object.entries(config.screens)) {
  test(`app route ${name} parses and roundtrips`, () => {
    const state = getStateFromPath(path, config);
    assert.equal(state.routes.at(-1).name, name);
    assert.equal(getPathFromState(state, config), `/${path}`);
  });
}

test('recovery parameters retain exact code and type values', () => {
  const state = getStateFromPath('reset-password?code=a%2Bb%25&type=recovery', config);
  assert.deepEqual(state.routes.at(-1).params, { code: 'a+b%', type: 'recovery' });
});

test('Unicode, literal plus, duplicate and empty query parameters survive', () => {
  const state = getStateFromPath('sign-up?name=%D0%90%D0%BB%D0%B8&tag=one&tag=two&empty=&plus=%2B', config);
  assert.deepEqual(state.routes.at(-1).params, { name: 'Али', tag: ['one', 'two'], empty: '', plus: '+' });
});

test('malformed percent input is processed by the installed parser', { timeout: 2000 }, () => {
  const start = performance.now();
  const state = getStateFromPath(`auth/callback?value=${'%FF'.repeat(20000)}`, config);
  assert.equal(state.routes.at(-1).name, 'AuthCallback');
  assert.ok(performance.now() - start < 1000, '60 KB malformed value should finish within one second');
});
