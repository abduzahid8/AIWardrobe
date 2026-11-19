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
  SHOWROOM_BOARD_WIDTH,
  SHOWROOM_BOARD_HEIGHT,
  SWAP_BOARD_WIDTH,
  SWAP_BOARD_HEIGHT,
  MAX_INPUT_IMAGE_BYTES,
  renderEditorialLookBoard,
  renderPortfolioComparisonSheet,
  renderVisualShowroomBoard,
  renderVisualSwapBoard,
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

test('stacks long title and subtitle vertically on 1024x1024 look boards with zero horizontal collision', async () => {
  const jpg = await syntheticPhoto('jpeg', '#887766', 120, 150);
  let capturedBaseSvg = '';
  const inspectingSharp = (input, opts) => {
    if (Buffer.isBuffer(input) && input.toString('utf8', 0, 40).includes('<svg')) {
      const text = input.toString('utf8');
      if (text.includes('S6-LIVE-06 AFTER')) capturedBaseSvg = text;
    }
    return sharp(input, opts);
  };
  const board = await renderEditorialLookBoard(
    [
      { id: 'o1', slot: 'outerwear', brand: 'Canali', price: 650, currency: 'USD', bytes: jpg },
      { id: 't1', slot: 'top', brand: 'Eton', price: 180, currency: 'USD', bytes: jpg },
      { id: 'b1', slot: 'bottom', brand: 'Incotex', price: 240, currency: 'USD', bytes: jpg },
      { id: 's1', slot: 'shoes', brand: 'Beckett Simonon', price: 229, currency: 'USD', bytes: jpg },
    ],
    {
      title: 'S6-LIVE-06 AFTER 1-TAP SHOE SWAP · Look 1 (Refined Evening Tailoring)',
      subtitle: 'Swapped Shoe: Beckett Simonon Dunham Brown Suede Derby ($229) · 13ms surgical swap',
      sharpImpl: inspectingSharp,
    },
  );
  assert.equal(board.ok, true);
  assert.ok(capturedBaseSvg.includes('y="42"'), 'title sits on top header baseline y=42');
  assert.ok(capturedBaseSvg.includes('x="36" y="66"'), 'subtitle stacks cleanly below title at y=66');
  assert.ok(!capturedBaseSvg.includes('x="988" y="54"'), 'legacy colliding same-baseline subtitle removed');
  const meta = await sharp(board.jpegBuffer).metadata();
  assert.equal(meta.width, 1024);
  assert.equal(meta.height, 1024);
});

test('renderVisualShowroomBoard renders a crisp 1600x1600 4-rack 32-SKU studio JPEG with bold T/L/B/S token overlays and AVIF/WebP/PNG/JPEG decoding', async () => {
  const [avifBuf, webpBuf, pngBuf, jpegBuf] = await Promise.all([
    syntheticPhoto('avif', '#EFECE6', 180, 240),
    syntheticPhoto('webp', '#263448', 180, 240),
    syntheticPhoto('png', '#4F5258', 180, 240),
    syntheticPhoto('jpeg', '#5B3D2E', 180, 140),
  ]);
  const makeSlotItems = (slot, buf, prefix) => Array.from({ length: 8 }, (_, idx) => ({
    id: `sku-${slot}-${idx + 1}`,
    macroCategory: slot,
    subtype: slot === 'top' ? 'dress-shirt' : slot === 'outerwear' ? 'blazer' : slot === 'bottom' ? 'tailored-trouser' : 'loafer',
    brand: `${prefix}House ${idx + 1}`,
    price: 90 + idx * 35,
    currency: 'USD',
    bytes: buf,
  }));

  const slotPools = {
    top: makeSlotItems('top', avifBuf, 'Top'),
    outerwear: makeSlotItems('outerwear', webpBuf, 'Coat'),
    bottom: makeSlotItems('bottom', pngBuf, 'Pant'),
    shoes: makeSlotItems('shoes', jpegBuf, 'Shoe'),
  };

  let capturedOverlaySvg = '';
  const inspectingSharp = (input, opts) => {
    const inst = sharp(input, opts);
    const origComposite = inst.composite.bind(inst);
    inst.composite = (layers) => {
      for (const layer of layers) {
        if (Buffer.isBuffer(layer?.input) && layer.input.toString('utf8', 0, 40).includes('<svg')) {
          capturedOverlaySvg = layer.input.toString('utf8');
        }
      }
      return origComposite(layers);
    };
    return inst;
  };

  const board = await renderVisualShowroomBoard(slotPools, {
    title: 'Milan Private Showroom',
    sharpImpl: inspectingSharp,
  });

  assert.equal(board.ok, true);
  assert.equal(board.width, SHOWROOM_BOARD_WIDTH);
  assert.equal(board.height, SHOWROOM_BOARD_HEIGHT);
  assert.equal(board.width, 1600);
  assert.equal(board.height, 1600);
  assert.equal(board.renderedCount, 32);
  assert.equal(board.totalRequested, 32);
  assert.ok(Buffer.isBuffer(board.jpegBuffer));
  assert.equal(board.sha256, createHash('sha256').update(board.jpegBuffer).digest('hex'));
  assert.equal(board.jpegBuffer[0], 0xff);
  assert.equal(board.jpegBuffer[1], 0xd8);
  assert.equal(board.jpegBuffer.readUInt16BE(board.jpegBuffer.length - 2), 0xffd9);

  // Verify Sharp metadata & warm #F6F5F2 canvas backdrop
  const meta = await sharp(board.jpegBuffer).metadata();
  assert.equal(meta.format, 'jpeg');
  assert.equal(meta.width, 1600);
  assert.equal(meta.height, 1600);
  const corner = await sharp(board.jpegBuffer).extract({ left: 2, top: 2, width: 1, height: 1 }).raw().toBuffer();
  assert.ok(Math.abs(corner[0] - 246) <= 4 && Math.abs(corner[1] - 245) <= 4 && Math.abs(corner[2] - 242) <= 4);

  // Verify exact token mapping & bold SVG overlays
  assert.equal(Object.keys(board.tokens).length, 32);
  assert.equal(board.tokens.T1.id, 'sku-top-1');
  assert.equal(board.tokens.T8.id, 'sku-top-8');
  assert.equal(board.tokens.L3.id, 'sku-outerwear-3');
  assert.equal(board.tokens.B5.id, 'sku-bottom-5');
  assert.equal(board.tokens.S2.id, 'sku-shoes-2');
  assert.equal(board.tokenMap.get('S8').id, 'sku-shoes-8');
  for (const tok of ['T1', 'T8', 'L3', 'B5', 'S2', 'S8']) {
    assert.ok(capturedOverlaySvg.includes(`>${tok}</text>`), `SVG must include crisp badge overlay for ${tok}`);
  }
});

test('renderVisualShowroomBoard honestly skips SSRF-blocked, HTTP-failed, and corrupted photos without emitting invalid tokens', async () => {
  const validJpg = await syntheticPhoto('jpeg', '#3A4B5C', 160, 220);
  const cache = createPhotoByteCache({ maxEntries: 32 });
  const fetchedUrls = [];
  const fetchImpl = async (url) => {
    fetchedUrls.push(url);
    if (url.includes('missing')) return new Response('not found', { status: 404 });
    if (url.includes('corrupt')) return new Response(Buffer.from('not-a-real-image-at-all-0123456789abcdef'), { status: 200 });
    return new Response(validJpg, { status: 200 });
  };

  const slotPools = {
    top: [
      { id: 't-ok-1', slot: 'top', brand: 'Eton', price: 180, currency: 'USD', imageUrl: 'https://static.zara.net/t1.jpg' },
      { id: 't-ssrf', slot: 'top', brand: 'Evil', price: 90, currency: 'USD', imageUrl: 'https://169.254.169.254/latest/meta-data/' },
      { id: 't-404', slot: 'top', brand: 'Gone', price: 90, currency: 'USD', imageUrl: 'https://static.zara.net/missing.jpg' },
      { id: 't-ok-2', slot: 'top', brand: 'COS', price: 95, currency: 'USD', imageUrl: 'https://static.zara.net/t2.jpg' },
    ],
    outerwear: [
      { id: 'l-corrupt', slot: 'outerwear', brand: 'BadImg', price: 320, currency: 'USD', imageUrl: 'https://static.zara.net/corrupt.jpg' },
      { id: 'l-ok-1', slot: 'outerwear', brand: 'Canali', price: 650, currency: 'USD', imageUrl: 'https://static.zara.net/l1.jpg' },
    ],
    bottom: [
      { id: 'b-ok-1', slot: 'bottom', brand: 'Incotex', price: 240, currency: 'USD', imageUrl: 'https://static.zara.net/b1.jpg' },
    ],
    shoes: [
      { id: 's-ok-1', slot: 'shoes', brand: 'Velasca', price: 290, currency: 'USD', imageUrl: 'https://static.zara.net/s1.jpg' },
    ],
  };

  const first = await renderVisualShowroomBoard(slotPools, { photoCache: cache, fetchImpl });
  assert.equal(first.ok, true);
  assert.equal(first.totalRequested, 8);
  assert.equal(first.renderedCount, 5, 'only 5 genuinely valid photos should render');
  assert.deepEqual(Object.keys(first.tokens).sort(), ['B1', 'L1', 'S1', 'T1', 'T2']);
  assert.equal(first.tokens.T1.id, 't-ok-1');
  assert.equal(first.tokens.T2.id, 't-ok-2');
  assert.equal(first.tokens.L1.id, 'l-ok-1');
  assert.ok(!fetchedUrls.some((u) => u.includes('169.254.169.254')), 'SSRF URL must never be fetched');
  assert.equal(first.skipped.length, 3);

  // Second render hits photoByteCache for all 5 valid URLs!
  const beforeFetches = fetchedUrls.length;
  const second = await renderVisualShowroomBoard(
    {
      top: [slotPools.top[0], slotPools.top[3]],
      outerwear: [slotPools.outerwear[1]],
      bottom: slotPools.bottom,
      shoes: slotPools.shoes,
    },
    { photoCache: cache, fetchImpl },
  );
  assert.equal(second.ok, true);
  assert.equal(second.cacheHits, 5);
  assert.equal(second.cacheMisses, 0);
  assert.equal(fetchedUrls.length, beforeFetches, 'cached showroom photos require 0 extra CDN fetches');
});

test('renderVisualSwapBoard renders a 1400x900 board with 3 locked garments on left (using photoCache) and R1..R6 candidates on right', async () => {
  const lockedJpg = await syntheticPhoto('jpeg', '#2B3A4D', 180, 260);
  const candJpg = await syntheticPhoto('webp', '#5D3D2A', 200, 160);
  const cache = createPhotoByteCache({ maxEntries: 32 });
  const fetches = [];
  const fetchImpl = async (url) => {
    fetches.push(url);
    if (url.includes('broken')) return new Response('gone', { status: 404 });
    return new Response(url.includes('cand') ? candJpg : lockedJpg, { status: 200 });
  };

  const lockedItems = [
    { id: 'lock-outer', macroCategory: 'outerwear', subtype: 'blazer', brand: 'Canali', price: 620, currency: 'USD', imageUrl: 'https://static.zara.net/lock-outer.jpg' },
    { id: 'lock-top', macroCategory: 'top', subtype: 'dress-shirt', brand: 'Eton', price: 185, currency: 'USD', imageUrl: 'https://static.zara.net/lock-top.jpg' },
    { id: 'lock-bottom', macroCategory: 'bottom', subtype: 'tailored-trouser', brand: 'Incotex', price: 245, currency: 'USD', imageUrl: 'https://static.zara.net/lock-bottom.jpg' },
  ];

  // Warm cache for the 3 locked items first (simulating post-look-board surgical swap!)
  await renderEditorialLookBoard(lockedItems, { photoCache: cache, fetchImpl });
  assert.equal(fetches.length, 3);

  const candidates = [
    { id: 'cand-1', macroCategory: 'shoes', subtype: 'loafer', brand: 'Velasca', price: 280, currency: 'USD', imageUrl: 'https://static.zara.net/cand-1.jpg' },
    { id: 'cand-broken', macroCategory: 'shoes', subtype: 'loafer', brand: 'Broken', price: 210, currency: 'USD', imageUrl: 'https://static.zara.net/cand-broken.jpg' },
    { id: 'cand-2', macroCategory: 'shoes', subtype: 'derby', brand: 'Carmina', price: 390, currency: 'USD', imageUrl: 'https://static.zara.net/cand-2.jpg' },
    { id: 'cand-3', macroCategory: 'shoes', subtype: 'oxford', brand: 'TLB Mallorca', price: 340, currency: 'USD', imageUrl: 'https://static.zara.net/cand-3.jpg' },
    { id: 'cand-4', macroCategory: 'shoes', subtype: 'chelsea', brand: 'Meermin', price: 260, currency: 'USD', imageUrl: 'https://static.zara.net/cand-4.jpg' },
    { id: 'cand-5', macroCategory: 'shoes', subtype: 'monk', brand: 'Berwick', price: 240, currency: 'USD', imageUrl: 'https://static.zara.net/cand-5.jpg' },
  ];

  let capturedOverlaySvg = '';
  const inspectingSharp = (input, opts) => {
    const inst = sharp(input, opts);
    const origComposite = inst.composite.bind(inst);
    inst.composite = (layers) => {
      for (const layer of layers) {
        if (Buffer.isBuffer(layer?.input) && layer.input.toString('utf8', 0, 40).includes('<svg')) {
          capturedOverlaySvg = layer.input.toString('utf8');
        }
      }
      return origComposite(layers);
    };
    return inst;
  };

  const swapBoard = await renderVisualSwapBoard(lockedItems, 'shoes', candidates, {
    photoCache: cache,
    fetchImpl,
    sharpImpl: inspectingSharp,
  });

  assert.equal(swapBoard.ok, true);
  assert.equal(swapBoard.width, SWAP_BOARD_WIDTH);
  assert.equal(swapBoard.height, SWAP_BOARD_HEIGHT);
  assert.equal(swapBoard.width, 1400);
  assert.equal(swapBoard.height, 900);
  assert.equal(swapBoard.renderedCount, 5, 'broken candidate skipped honestly; 5 rendered');
  assert.equal(swapBoard.cacheHits, 3, 'all 3 locked items hit photoCache at 0ms CDN overhead');
  assert.deepEqual(Object.keys(swapBoard.candidateTokens), ['R1', 'R2', 'R3', 'R4', 'R5']);
  assert.equal(swapBoard.candidateTokens.R1.id, 'cand-1');
  assert.equal(swapBoard.candidateTokens.R2.id, 'cand-2');
  assert.equal(swapBoard.candidateTokens.R5.id, 'cand-5');
  assert.equal(swapBoard.lockedSlotStatus.shoes.status, 'target_slot');
  assert.equal(swapBoard.lockedSlotStatus.outerwear.status, 'rendered');
  assert.ok(capturedOverlaySvg.includes('SWAP TARGET'));
  for (const tok of ['R1', 'R2', 'R3', 'R4', 'R5']) {
    assert.ok(capturedOverlaySvg.includes(`>${tok}</text>`), `Swap board SVG must include bold badge ${tok}`);
  }
  const meta = await sharp(swapBoard.jpegBuffer).metadata();
  assert.equal(meta.format, 'jpeg');
  assert.equal(meta.width, 1400);
  assert.equal(meta.height, 900);
});
