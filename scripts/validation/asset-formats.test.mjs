import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../assets', import.meta.url));
function images(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => entry.isDirectory()
    ? images(join(dir, entry.name)) : /\.(png|jpe?g)$/i.test(entry.name) ? [join(dir, entry.name)] : []);
}
test('PNG and JPEG asset names match their actual image encoding', () => {
  for (const path of images(root)) {
    const bytes = readFileSync(path);
    const valid = extname(path).toLowerCase() === '.png'
      ? bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))
      : bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    assert.ok(valid, `Image encoding does not match extension: ${path}`);
  }
});
