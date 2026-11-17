import test from 'node:test';
import assert from 'node:assert/strict';
import { STYLIST6_DEV_ENGINE_VERSION, isCurrentStylistBridgeHealth } from './engine_identity.mjs';

test('current development bridge is accepted only with matching version', () => {
  assert.equal(isCurrentStylistBridgeHealth({
    ok: true, engine: STYLIST6_DEV_ENGINE_VERSION, references: 46,
  }), true);
  assert.equal(isCurrentStylistBridgeHealth({
    ok: true, engine: 'stylist-6-dev-local-20261111-r2', references: 46,
  }), false);
  assert.equal(isCurrentStylistBridgeHealth({
    ok: false, engine: STYLIST6_DEV_ENGINE_VERSION,
  }), false);
  assert.equal(isCurrentStylistBridgeHealth({ ok: true, engine: 'unrelated' }), false);
  assert.equal(isCurrentStylistBridgeHealth(null), false);
  assert.equal(isCurrentStylistBridgeHealth({}), false);
});
