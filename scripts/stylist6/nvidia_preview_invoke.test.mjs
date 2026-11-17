import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const script = path.join(root, 'scripts/stylist6/nvidia_preview_invoke.mjs');

function run(args = [], env = {}) {
  return spawnSync(process.execPath, [script, ...args], {
    encoding: 'utf8',
    env: { ...process.env, STYLIST_NODE_SHARED_SECRET: '', NVIDIA_API_KEY: '', ...env },
  });
}

test('refuses without secret before any network', () => {
  const r = run(['--url', 'https://preview.example.com/api/stylist6-vision-test', '--confirm-spend']);
  assert.equal(r.status, 2);
  assert.match(r.stderr, /STYLIST_NODE_SHARED_SECRET/);
});

test('refuses without spend approval and flag', () => {
  const r = run(['--url', 'https://preview.example.com/api/stylist6-vision-test'],
    { STYLIST_NODE_SHARED_SECRET: 'x'.repeat(32) });
  assert.equal(r.status, 2);
  assert.match(r.stderr, /NVIDIA_SPEND_APPROVED/);
});

test('refuses non-HTTPS or off-path URLs', () => {
  const env = { STYLIST_NODE_SHARED_SECRET: 'x'.repeat(32), NVIDIA_SPEND_APPROVED: '1' };
  const r = run(['--url', 'http://evil.example.com/steal', '--confirm-spend'], env);
  assert.equal(r.status, 2);
  assert.match(r.stderr, /--url https/);
});

test('refuses without verified reference bytes', () => {
  const env = {
    STYLIST_NODE_SHARED_SECRET: 'x'.repeat(32),
    NVIDIA_SPEND_APPROVED: '1',
    NVIDIA_REFERENCE_RIGHTS_CONFIRMED: '1',
  };
  const r = run([
    '--url', 'https://preview.example.com/api/stylist6-vision-test',
    '--ref', '/nonexistent.jpg', '--products', 'https://example.com/p.jpg',
    '--confirm-spend',
  ], env);
  assert.equal(r.status, 2);
  assert.match(r.stderr, /--ref verified reference JPEG/);
});
