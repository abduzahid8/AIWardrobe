import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import sharp from 'sharp';
import {
  buildHumanBlindPackage,
  isAllowlistedImageUrl,
  scoreHumanBlindSubmissions,
  selectComparableThreeLookScenarios,
} from './build_human_blind_package.mjs';

test('isAllowlistedImageUrl blocks unknown hosts and non-HTTPS URLs', () => {
  assert.equal(isAllowlistedImageUrl('https://static.zara.net/photos/shirt.jpg'), true);
  assert.equal(isAllowlistedImageUrl('http://static.zara.net/photos/shirt.jpg'), false);
  assert.equal(isAllowlistedImageUrl('https://evil.example.com/shirt.jpg'), false);
});

test('buildHumanBlindPackage builds a randomized >=30-scenario blind package and scores genuine submissions', async () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'aiw-human-blind-'));
  try {
    const jpegBytes = await sharp({
      create: { width: 64, height: 64, channels: 3, background: { r: 30, g: 50, b: 90 } },
    }).jpeg({ quality: 80 }).toBuffer();

    const makeIds = (prefix) => ['t1', 'o1', 'b1', 's1'].map((x) => `${prefix}-${x}`);
    const catalog = {};
    for (const prefix of ['l1', 'l2', 'l3', 'r1', 'r2', 'r3']) {
      for (const [slot, code] of [['top', 't1'], ['outerwear', 'o1'], ['bottom', 'b1'], ['shoes', 's1']]) {
        const id = `${prefix}-${code}`;
        catalog[id] = {
          id, slot, name: `${slot} ${id}`, brand: 'Test', price: 100000, currency: 'UZS',
          imageUrl: `https://static.zara.net/test/${id}.jpg`,
        };
      }
    }
    const results = Array.from({ length: 30 }, (_, i) => ({
      scenario: { id: i < 7 ? `S6-H0${i + 1}` : `S6-${i}`, style: 'classic', occasion: 'Work', weather: { temp: 18 } },
      arms: {
        stylist5: [{ ids: makeIds('l1') }, { ids: makeIds('l2') }, { ids: makeIds('l3') }],
        stylist6Fashion: [{ ids: makeIds('r1') }, { ids: makeIds('r2') }, { ids: makeIds('r3') }],
      },
    }));
    const replayPath = path.join(dir, 'replay.json');
    writeFileSync(replayPath, JSON.stringify({ catalog, results }));

    assert.equal(selectComparableThreeLookScenarios({ results }).length, 30);

    const outDir = path.join(dir, 'pkg');
    const summary = await buildHumanBlindPackage({
      replayPath,
      outDir,
      cacheDir: path.join(dir, 'cache'),
      minScenarios: 30,
      fetchImpl: async () => new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } }),
    });
    assert.equal(summary.scenarioCount, 30);
    assert.equal(summary.heldOutCount, 7);

    const html = readFileSync(path.join(outDir, 'index.html'), 'utf8');
    assert.doesNotMatch(html, /stylist6Fashion|stylist5/);

    const keyDoc = JSON.parse(readFileSync(path.join(outDir, 'blind_key.json'), 'utf8'));
    const mkSub = (reviewer, pickS6Count) => {
      const values = { reviewer };
      let i = 0;
      for (const [id, info] of Object.entries(keyDoc.scenarios)) {
        const s6Label = info.A === 'stylist6Fashion' ? 'A' : 'B';
        const s5Label = s6Label === 'A' ? 'B' : 'A';
        values[`${id}.preference`] = i < pickS6Count ? s6Label : s5Label;
        i += 1;
      }
      return { packageSha256: keyDoc.packageSha256, reviewer, values };
    };

    const scored = scoreHumanBlindSubmissions({
      keyDoc,
      submissions: [mkSub('rev_1', 21), mkSub('rev_2', 21), mkSub('rev_3', 21)],
    });
    assert.equal(scored.s6Wins, 21);
    assert.equal(scored.winRate, 0.7);
    assert.equal(scored.meets70PercentThreshold, true);
    assert.equal(scored.humanBlind.scenarios.length, 30);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
