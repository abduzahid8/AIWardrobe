/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const { PNG } = require('pngjs');
const jpeg = require('jpeg-js');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/_shared/tryOnComposite.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
const api = {};
new Function('require', 'exports', compiled)(name => require(name.startsWith('npm:') ? name.slice(4).replace(/@[^@/]+$/, '') : name), api);
const { composeTryOnImages } = api;
function png(width, height, rgba) {
  const image = new PNG({ width, height });
  for (let i = 0; i < image.data.length; i += 4) image.data.set(rgba, i);
  return `data:image/png;base64,${PNG.sync.write(image).toString('base64')}`;
}
const read = result => PNG.sync.read(Buffer.from(result.split(',')[1], 'base64'));
const pixel = (image, x, y) => [...image.data.subarray((y * image.width + x) * 4, (y * image.width + x) * 4 + 4)];
test('preserves the person AND garment references in separate panels without browser canvas APIs', () => {
  const image = read(composeTryOnImages(png(1, 1, [255, 0, 0, 255]), png(1, 1, [0, 0, 255, 255]), 8, 4));
  expect([image.width, image.height]).toEqual([8, 4]);
  expect(pixel(image, 1, 2)).toEqual([255, 0, 0, 255]);
  expect(pixel(image, 6, 2)).toEqual([0, 0, 255, 255]);
});
test('fits aspect ratio on white and correctly blends transparent cutouts', () => {
  const image = read(composeTryOnImages(png(1, 2, [255, 0, 0, 255]), png(1, 1, [0, 0, 0, 0]), 8, 4));
  expect(pixel(image, 0, 1)).toEqual([255, 255, 255, 255]);
  expect(pixel(image, 1, 1)).toEqual([255, 0, 0, 255]);
  expect(pixel(image, 6, 1)).toEqual([255, 255, 255, 255]);
});
test('supports JPEG input using the bounded server decoder', () => {
  const encoded = jpeg.encode({ width: 1, height: 1, data: Buffer.from([255, 0, 0, 255]) }, 95).data.toString('base64');
  const image = read(composeTryOnImages(encoded, png(1, 1, [0, 0, 255, 255]), 8, 4));
  expect(pixel(image, 1, 1)[0]).toBeGreaterThan(240);
  expect(pixel(image, 6, 1)[2]).toBe(255);
});
test('invalid input never quietly drops the garment reference', () => {
  expect(() => composeTryOnImages(png(1, 1, [0, 0, 0, 255]), 'not-an-image')).toThrow();
});
test('rejects excessive PNG dimensions before decoder allocation', () => {
  const encoded = Buffer.from(png(1, 1, [0, 0, 0, 255]).split(',')[1], 'base64');
  encoded.writeUInt32BE(100_000, 16);
  encoded.writeUInt32BE(100_000, 20);
  expect(() => composeTryOnImages(encoded.toString('base64'), png(1, 1, [0, 0, 0, 255]))).toThrow('resolution');
});
