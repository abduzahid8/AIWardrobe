import test from 'node:test';
import assert from 'node:assert/strict';
import { mapDevVisualStatus, referenceFileFor, STYLIST6_DEV_ENGINE_VERSION } from './dev_bridge.mjs';

test('visual statuses distinguish applied, unchanged, abstained and not-run decks', () => {
  assert.equal(mapDevVisualStatus({ diagnostics: { visualVerdict: 'NOT_RUN' }, outfits: [] }), 'NVIDIA_NOT_RUN');
  assert.equal(mapDevVisualStatus({ diagnostics: { visualVerdict: 'ABSTAINED' }, outfits: [] }), 'NVIDIA_ABSTAINED');
  assert.equal(
    mapDevVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ visualEvidence: ['a', 'b', 'c'] }] }),
    'NVIDIA_APPLIED',
  );
  assert.equal(
    mapDevVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ id: 's6_x' }] }),
    'NVIDIA_EVALUATED_NO_CHANGE',
  );
  assert.equal(mapDevVisualStatus(null), 'NVIDIA_NOT_RUN');
});

test('reference admission requires a real local photo, not a metadata entry', () => {
  // rt_001 has a metadata entry and a matching local lookbook photo.
  assert.ok(referenceFileFor('rt_001')?.endsWith('.jpg'));
  // Unknown ids (metadata without bytes) fail admission honestly.
  assert.equal(referenceFileFor('rt_999'), null);
  assert.equal(referenceFileFor(''), null);
});

test('local bridge engine identity is bumped after screenshot-driven remediation',()=>{
  assert.equal(STYLIST6_DEV_ENGINE_VERSION,'stylist-6-dev-local-20261112-r3');
});
