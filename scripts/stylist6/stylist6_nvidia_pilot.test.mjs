import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { loadPilotCase } from './stylist6_nvidia_pilot.mjs';

const cli = new URL('./stylist6_nvidia_pilot.mjs', import.meta.url);
const digest = (value) => createHash('sha256').update(value).digest('hex');

function makeFixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stylist6-pilot-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.mkdirSync(path.join(dir, 'photos'));
  fs.mkdirSync(path.join(dir, 'references'));
  const write = (name, value) => {
    const body = JSON.stringify(value);
    fs.writeFileSync(path.join(dir, name), body);
    return digest(body);
  };
  const jpeg = Buffer.alloc(256, 7);
  Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0, 8, 8, 0, 16, 0, 16, 3]).copy(jpeg);
  jpeg.writeUInt16BE(0xffd9, jpeg.length - 2);
  const hash = digest(jpeg);
  const photo = 'photos/' + hash + '.jpg';
  fs.writeFileSync(path.join(dir, photo), jpeg);
  fs.writeFileSync(path.join(dir, 'references/rt_001.jpg'), jpeg);
  write('references/hashes.json', { rt_001: hash });
  const scenario = {
    id: 'Q01', reference: 'rt_001', style: 'classic', occasion: 'Work', brief: 'Soft tailoring',
    budget: { amount: 100, currency: 'USD' },
  };
  const items = [
    ['shirt1', 'White shirt', 'tops'], ['shirt2', 'Blue shirt', 'tops'],
    ['shirt3', 'Cream shirt', 'tops'], ['shirt4', 'Grey shirt', 'tops'],
    ['layer', 'Navy blazer', 'tops'], ['bottom', 'Grey trousers', 'bottoms'], ['shoes', 'Brown loafer', 'shoes'],
  ];
  const inventorySha256 = write('inventory.json', items.map(([id, name, category]) => ({
    row: { id, name, category, is_active: true, price: 25, currency: 'USD',
      image_url: 'https://cdn.example.com/' + id, source_url: 'https://shop.example.com/' + id },
    photo, photoSha256: hash,
  })));
  const scenariosSha256 = write('scenarios.json', [scenario]);
  write('paired_critic_queue.json', { inventorySha256, scenariosSha256, cases: [{
    scenario, candidates: [1, 2, 3, 4].map((n) => ({
      id: 'Q01-0' + n, origin: n === 1 ? 'v5' : 's6_' + n, items: ['shirt' + n, 'layer', 'bottom', 'shoes'],
    })),
  }] });
  return { dir, photo };
}

test('dry run pins complete, budget-eligible candidate photos and detects modified media', (t) => {
  const { dir, photo } = makeFixture(t);
  const result = spawnSync(process.execPath, [cli.pathname, '--organizer', dir, '--scenario', 'Q01'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const receipt = JSON.parse(result.stdout);
  assert.equal(receipt.status, 'NOT_RUN');
  assert.equal(receipt.providerCalls, 0);
  assert.equal(receipt.plan.maxProviderCalls, 4);
  assert.equal(receipt.approvalSha256, loadPilotCase(dir, 'Q01').approvalSha256);
  const file = path.join(dir, photo);
  const bytes = fs.readFileSync(file);
  bytes[50] = 19;
  fs.writeFileSync(file, bytes);
  assert.throws(() => loadPilotCase(dir, 'Q01'), /image_digest_mismatch/);
});

test('an existing claim or a mismatched approval is refused before dispatch', (t) => {
  const { dir } = makeFixture(t);
  const { approvalSha256 } = loadPilotCase(dir, 'Q01');
  const preload = path.join(dir, 'no-network.mjs');
  fs.writeFileSync(preload, 'globalThis.fetch = () => { process.exit(99); };');
  const env = { ...process.env, NVIDIA_SPEND_APPROVED: '1', NVIDIA_REFERENCE_RIGHTS_CONFIRMED: '1',
    NVIDIA_API_KEY: 'test-key-never-use-network' };
  const claim = path.join(dir, 'nvidia-' + approvalSha256 + '.json');
  fs.writeFileSync(claim, 'prior claim');
  for (const [approval, reason] of [['wrong', 'pilot_approval_required'], [approvalSha256, 'pilot_already_claimed']]) {
    const result = spawnSync(process.execPath, [
      '--import', preload, cli.pathname, '--organizer', dir, '--scenario', 'Q01', '--approve-plan', approval,
    ], { encoding: 'utf8', env });
    assert.equal(result.status, 2, result.stderr);
    assert.match(result.stderr, new RegExp(reason));
    assert.equal(fs.readFileSync(claim, 'utf8'), 'prior claim');
  }
});
