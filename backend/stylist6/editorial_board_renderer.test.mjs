import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import {
  EDITORIAL_BOARD_RENDERER_VERSION,
  LOOK_BOARD_WIDTH,
  LOOK_BOARD_HEIGHT,
  PORTFOLIO_SHEET_WIDTH,
  PORTFOLIO_SHEET_HEIGHT,
  MAX_INPUT_IMAGE_BYTES,
  renderEditorialLookBoard,
  renderPortfolioComparisonSheet,
  createPhotoByteCache,
  validateEditorialImageUrl,
  decodeNormalizedPhotoBytes,
} from './editorial_board_renderer.mjs';

async function syntheticPhoto(format, hexColor, w = 220, h = 300) {
  const base = sharp({ create: { width: w, height: h, channels: 3, background: hexColor } });
  if (format === 'png') return base.png().toBuffer();
  if (format === 'webp') return base.webp({ quality: 80 }).toBuffer();
  if (format === 'avif') return base.avif({ quality: 50 }).toBuffer();
  return base.jpeg({ quality: 80 }).toBuffer();
}

test('renderEditorialLookBoard renders a valid 1024x1024 studio JPEG decoding JPEG, PNG, WebP, and AVIF', async () => {
  assert.match(EDITORIAL_BOARD_RENDERER_VERSION, /editorial-board-renderer/);
  const [outerBytes, topBytes, bottomBytes, shoesBytes] = await Promise.all([
    syntheticPhoto('avif', '#8B6F4E', 240, 360),
    syntheticPhoto('webp', '#F2EDE4', 220, 260),
    syntheticPhoto('png', '#363C45', 220, 360),
    syntheticPhoto('jpeg', '#4E3426', 240, 180),
  ]);

  const items = [
    { id: 'sku-outer', slot: 'outerwear', brand: 'Loro Piana', price: 1250, currency: 'USD', bytes: outerBytes },
    { id: 'sku-top', slot: 'top', brand: 'COS', price: 95, currency: 'USD', bytes: topBytes },
    { id: 'sku-bottom', slot: 'bottom', brand: 'Massimo Dutti', price: 140, currency: 'USD', bytes: bottomBytes },
    { id: 'sku-shoes', slot: 'shoes', brand: 'Velasca', price: 320, currency: 'USD', bytes: shoesBytes },
  ];

  const board = await renderEditorialLookBoard(items, {
    title: 'Autumn Tailored Ease',
    subtitle: 'CLASSIC · WORK · 15°C',
  });

  assert.equal(board.ok, true);
  assert.equal(board.width, LOOK_BOARD_WIDTH);
  assert.equal(board.height, LOOK_BOARD_HEIGHT);
  assert.equal(board.mediaType, 'image/jpeg');
  assert.equal(board.allSlotsRendered, true);
  assert.ok(Buffer.isBuffer(board.jpegBuffer));
  assert.equal(board.sha256, createHash('sha256').update(board.jpegBuffer).digest('hex'));
  assert.equal(board.jpegBuffer[0], 0xff);
  assert.equal(board.jpegBuffer[1], 0xd8);
  assert.equal(board.jpegBuffer.readUInt16BE(board.jpegBuffer.length - 2), 0xffd9);

  // Verify decoded metadata and warm #F6F5F2 backdrop pixel via Sharp
  const meta = await sharp(board.jpegBuffer).metadata();
  assert.equal(meta.format, 'jpeg');
  assert.equal(meta.width, 1024);
  assert.equal(meta.height, 1024);

  const corner = await sharp(board.jpegBuffer)
    .extract({ left: 2, top: 2, width: 1, height: 1 })
    .raw()
    .toBuffer();
  // #F6F5F2 is (246, 245, 242); allow small JPEG quantization tolerance
  assert.ok(Math.abs(corner[0] - 246) <= 4 && Math.abs(corner[1] - 245) <= 4 && Math.abs(corner[2] - 242) <= 4);

  // Verify every input format was transparently decoded
  assert.equal(board.slotStatus.outerwear.status, 'rendered');
  assert.equal(board.slotStatus.outerwear.sourceFormat, 'avif');
  assert.equal(board.slotStatus.top.sourceFormat, 'webp');
  assert.equal(board.slotStatus.bottom.sourceFormat, 'png');
  assert.equal(board.slotStatus.shoes.sourceFormat, 'jpeg');
});

test('renderPortfolioComparisonSheet renders valid 1800x1000 high-res JPEGs for 3-look and 6-look portfolios', async () => {
  const palette = ['#2D384C', '#EFECE6', '#56595F', '#3F2A1E'];
  const bufs = await Promise.all(palette.map((c) => syntheticPhoto('jpeg', c, 180, 240)));
  const makeLook = (idx) => ({
    id: `look_${idx}`,
    title: `Direction ${idx + 1}`,
    editorialScore: 8.4 - idx * 0.2,
    items: ['outerwear', 'top', 'bottom', 'shoes'].map((slot, sIdx) => ({
      id: `item_${idx}_${sIdx}`,
      macroCategory: slot,
      bytes: bufs[(sIdx + idx) % bufs.length],
    })),
  });

  // 3-look flagship 3-column layout
  const threeSheet = await renderPortfolioComparisonSheet([makeLook(0), makeLook(1), makeLook(2)], {
    title: 'Classic Date Night Curation',
    subtitle: '3 DISTINCT DIRECTIONS',
  });
  assert.equal(threeSheet.ok, true);
  assert.equal(threeSheet.width, PORTFOLIO_SHEET_WIDTH);
  assert.equal(threeSheet.height, PORTFOLIO_SHEET_HEIGHT);
  assert.equal(threeSheet.mediaType, 'image/jpeg');
  assert.equal(threeSheet.allLooksRendered, true);
  assert.deepEqual(threeSheet.lookTokens, ['L00', 'L01', 'L02']);
  const meta3 = await sharp(threeSheet.jpegBuffer).metadata();
  assert.equal(meta3.format, 'jpeg');
  assert.equal(meta3.width, 1800);
  assert.equal(meta3.height, 1000);

  // 6-look 2x3 comparison layout
  const sixSheet = await renderPortfolioComparisonSheet(Array.from({ length: 6 }, (_, i) => makeLook(i)), {
    anonymizeLabels: true,
  });
  assert.equal(sixSheet.ok, true);
  assert.equal(sixSheet.width, 1800);
  assert.equal(sixSheet.height, 1000);
  assert.equal(sixSheet.lookTokens.length, 6);
  const meta6 = await sharp(sixSheet.jpegBuffer).metadata();
  assert.equal(meta6.width, 1800);
  assert.equal(meta6.height, 1000);
});

test('SSRF host validation blocks private IPs, HTTP, ports, credentials, and unallowlisted hosts with zero fetch', async () => {
  const blockedUrls = [
    'http://static.zara.net/photo.jpg',
    'https://127.0.0.1/secret.jpg',
    'https://169.254.169.254/latest/meta-data/',
    'https://localhost/admin.jpg',
    'https://10.0.0.1/internal.jpg',
    'https://192.168.1.5/nas.jpg',
    'https://static.zara.net:8443/photo.jpg',
    'https://user:pass@static.zara.net/photo.jpg',
    'https://evil.example.com/phish.jpg',
  ];
  for (const url of blockedUrls) {
    assert.equal(validateEditorialImageUrl(url).ok, false, `must block ${url}`);
  }
  assert.equal(validateEditorialImageUrl('https://static.zara.net/item/1.jpg').ok, true);

  let fetchCalls = 0;
  const board = await renderEditorialLookBoard(
    [
      { id: '1', slot: 'top', imageUrl: 'https://169.254.169.254/latest/meta-data/' },
      { id: '2', slot: 'bottom', imageUrl: 'https://evil.example.com/a.jpg' },
      { id: '3', slot: 'shoes', imageUrl: 'http://static.zara.net/c.jpg' },
    ],
    {
      requireAllPhotos: true,
      fetchImpl: async () => { fetchCalls++; throw new Error('must_never_fetch'); },
    },
  );
  assert.equal(fetchCalls, 0);
  assert.equal(board.ok, false);
  assert.equal(board.jpegBuffer, null);
  assert.equal(board.slotStatus.top.status, 'ssrf_blocked');
  assert.equal(board.slotStatus.bottom.status, 'host_not_allowlisted');
  assert.equal(board.slotStatus.shoes.status, 'unsafe_url');
});

test('bounded LRU photo byte cache serves 3 locked items with 0ms extra CDN fetch on surgical swap', async () => {
  const jpg = await syntheticPhoto('jpeg', '#607080', 180, 220);
  const swapJpg = await syntheticPhoto('jpeg', '#4A3728', 180, 220);
  const cache = createPhotoByteCache({ maxEntries: 16, maxBytes: 2 * 1024 * 1024 });

  const fetches = [];
  const fetchImpl = async (url) => {
    fetches.push(url);
    return new Response(url.includes('swap') ? swapJpg : jpg, { status: 200 });
  };

  const baseItems = [
    { id: 'o1', slot: 'outerwear', imageUrl: 'https://static.zara.net/o1.jpg' },
    { id: 't1', slot: 'top', imageUrl: 'https://static.zara.net/t1.jpg' },
    { id: 'b1', slot: 'bottom', imageUrl: 'https://static.zara.net/b1.jpg' },
    { id: 's1', slot: 'shoes', imageUrl: 'https://static.zara.net/s1.jpg' },
  ];

  const round1 = await renderEditorialLookBoard(baseItems, { photoCache: cache, fetchImpl, requireAllPhotos: true });
  assert.equal(round1.ok, true);
  assert.equal(fetches.length, 4);
  assert.equal(round1.cacheHits, 0);
  assert.equal(round1.cacheMisses, 4);

  // Surgical swap: lock outerwear/top/bottom, swap ONLY shoes
  const swappedItems = [
    ...baseItems.slice(0, 3),
    { id: 's2', slot: 'shoes', imageUrl: 'https://static.zara.net/s2-swap.jpg' },
  ];
  const round2 = await renderEditorialLookBoard(swappedItems, { photoCache: cache, fetchImpl, requireAllPhotos: true });
  assert.equal(round2.ok, true);
  assert.equal(fetches.length, 5, 'only the 1 newly swapped shoe should fetch from CDN');
  assert.equal(round2.cacheHits, 3, '3 locked items served straight from photo cache');
  assert.equal(round2.cacheMisses, 1);
  assert.notEqual(round2.sha256, round1.sha256);

  // Verify LRU eviction under tight byte cap
  const tinyCache = createPhotoByteCache({ maxEntries: 10, maxBytes: 100 * 1024 });
  const bigBuf = Buffer.alloc(60 * 1024, 1);
  assert.equal(tinyCache.set('k1', bigBuf), true);
  assert.equal(tinyCache.set('k2', bigBuf), true);
  assert.equal(tinyCache.has('k1'), false, 'oldest evicted when byte cap exceeded');
  assert.equal(tinyCache.has('k2'), true);
  assert.equal(tinyCache.stats().evictions, 1);
});

test('enforces byte cap and safely escapes XML/SVG injection in brand/title labels', async () => {
  await assert.rejects(
    () => decodeNormalizedPhotoBytes(Buffer.alloc(MAX_INPUT_IMAGE_BYTES + 1)),
    (err) => err.code === 'oversize',
  );

  const jpg = await syntheticPhoto('jpeg', '#998877', 120, 150);
  const board = await renderEditorialLookBoard(
    [
      { id: '1', slot: 'top', brand: '<script>alert(1)</script> & Co', price: 89, currency: 'USD', bytes: jpg },
      { id: '2', slot: 'bottom', brand: 'https://evil.example/x "Brand"', price: 120, currency: 'EUR', bytes: jpg },
      { id: '3', slot: 'shoes', brand: 'Atelier & Sons', price: 240, currency: 'USD', bytes: jpg },
    ],
    { title: 'Look <1> & "Date" https://secret.example/token' },
  );
  assert.equal(board.ok, true);
  assert.equal(board.slotStatus.outerwear.status, 'absent');
  const meta = await sharp(board.jpegBuffer).metadata();
  assert.equal(meta.width, 1024);
  assert.equal(meta.height, 1024);
});
