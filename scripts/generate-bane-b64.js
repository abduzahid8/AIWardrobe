#!/usr/bin/env node
/**
 * Converts bane_mannequin.glb to a base64 data-URI TypeScript constant.
 * Run once: node scripts/generate-bane-b64.js
 */
const fs   = require('fs');
const path = require('path');

const root   = path.join(__dirname, '..');
const input  = path.join(root, 'assets', 'models', 'bane_mannequin.glb');
const output = path.join(root, 'features', 'try-on', 'utils', 'baneModelB64.ts');

if (!fs.existsSync(input)) {
  console.error('ERROR: bane_mannequin.glb not found at', input);
  process.exit(1);
}

console.log('Reading GLB…');
const buf    = fs.readFileSync(input);
const b64    = buf.toString('base64');
const dataUri = `data:model/gltf-binary;base64,${b64}`;

const ts = `// AUTO-GENERATED — run scripts/generate-bane-b64.js to regenerate
// Source: assets/models/bane_mannequin.glb (${(buf.length / 1024 / 1024).toFixed(1)} MB)
export const BANE_MODEL_DATA_URI = '${dataUri}';
`;

fs.writeFileSync(output, ts, 'utf8');
console.log('Written to', output, `(${(ts.length / 1024 / 1024).toFixed(1)} MB)`);
