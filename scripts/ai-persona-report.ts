/**
 * ai-persona-report.ts — visual proof for the multi-account AI e2e.
 *
 * Runs the SAME personaHarness used by __tests__/e2e/aiPersonas.e2e.test.ts
 * (same personas, same production modules, same assertions), renders an HTML
 * report, and screenshots it with Playwright Chromium.
 *
 * Usage:  npx tsx scripts/ai-persona-report.ts
 * Output: /tmp/ai-personas/report.html + report.png + persona-*.png
 */
import { writeFileSync, mkdirSync } from 'fs';
import { chromium } from 'playwright-core';
import { runAllPersonas, type PersonaResult } from '../__tests__/e2e/personaHarness';

const OUT = '/tmp/ai-personas';

const esc = (s: string) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const SLOT_LABEL: Record<string, string> = { outerwear: 'L', top: 'T', bottom: 'B', shoes: 'S' };
const SLOT_NAME: Record<string, string> = { outerwear: 'Layer', top: 'Top', bottom: 'Bottom', shoes: 'Shoes' };

function card(item: PersonaResult['deck'][number][number]): string {
  const initial = esc((item.name || '?').trim().charAt(0).toUpperCase() || '?');
  return `<div class="item">
    <div class="thumb"><span class="mono">${initial}</span>${
      item.imageUrl
        ? `<img src="${esc(item.imageUrl)}" alt="" loading="lazy" onerror="this.remove()" />`
        : ''
    }<span class="slot">${SLOT_LABEL[item.slot] || '?'}</span></div>
    <div class="iname">${esc(item.name)}</div>
    <div class="ibrand">${esc(item.brand || '')} · ${esc(SLOT_NAME[item.slot] || item.slot)}</div>
    <div class="score"><i style="width:${Math.round(item.score * 100)}%"></i></div>
  </div>`;
}

function personaSection(r: PersonaResult, idx: number): string {
  const p = r.persona;
  const outfits = r.deck
    .map(
      (o, k) => `<div class="outfit"><div class="ono">Look ${k + 1}</div><div class="items">${o
        .map(card)
        .join('')}</div></div>`,
    )
    .join('');
  const checks = r.checks
    .map(
      (c) =>
        `<div class="check ${c.pass ? 'pass' : 'fail'}"><span class="pill">${
          c.pass ? 'PASS' : 'FAIL'
        }</span><div><b>${esc(c.label)}</b><span>${esc(c.detail)}</span></div></div>`,
    )
    .join('');
  return `<section class="persona" id="persona-${idx}">
    <div class="phead"><div class="avatar">${esc(p.account.charAt(0))}</div>
      <div><h2>${esc(p.account)}</h2>
      <div class="meta"><span class="chip style">${esc(p.styleLabel)}</span>
      <span class="chip">${esc(p.occasion)}</span>
      <span class="chip">${p.weather.temp}°C ${esc(p.weather.condition)}</span>
      <span class="chip ${p.mode}">${p.mode === 'shop' ? 'New clothing' : 'From wardrobe'}</span>
      <span class="chip ghost">pool ${r.poolSize} · ${r.layered ? 'layered' : 'unlayered'}</span></div></div></div>
    <div class="outfits">${outfits}</div>
    <div class="checks">${checks}</div>
  </section>`;
}

async function main() {
  const results = runAllPersonas();
  const passed = results.flatMap((r) => r.checks).filter((c) => c.pass).length;
  const total = results.flatMap((r) => r.checks).length;
  const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
  <style>
    * { box-sizing: border-box; font-family: -apple-system, 'SF Pro Text', Inter, Helvetica, Arial, sans-serif; }
    body { margin: 0; background: #EEF4FF; color: #0A1931; }
    header { background: #0A1931; color: #fff; padding: 36px 40px; }
    header h1 { margin: 0 0 8px; font-size: 30px; }
    header p { margin: 4px 0; color: #B9C6DD; font-size: 14px; }
    .score-top { display: inline-block; margin-top: 12px; background: #10B981; color: #fff; font-weight: 800; padding: 8px 16px; border-radius: 999px; }
    main { padding: 28px 40px 60px; max-width: 1240px; margin: 0 auto; }
    .persona { background: #fff; border-radius: 24px; padding: 28px; margin-bottom: 28px; box-shadow: 0 10px 30px rgba(23,58,101,.08); }
    .phead { display: flex; gap: 16px; align-items: center; margin-bottom: 20px; }
    .avatar { width: 56px; height: 56px; border-radius: 50%; background: #254F86; color: #fff; font-size: 26px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
    h2 { margin: 0 0 8px; font-size: 21px; }
    .meta { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { font-size: 12px; font-weight: 700; background: #EEF4FF; border: 1px solid #D8E3F5; padding: 4px 12px; border-radius: 999px; }
    .chip.style { background: #0A1931; color: #fff; border-color: #0A1931; }
    .chip.shop { background: #E6F4FF; border-color: #B9D9F5; }
    .chip.wardrobe { background: #F3EFE6; border-color: #E2D9C3; }
    .chip.ghost { color: #64748B; }
    .outfits { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
    .outfit { background: #F8FAFF; border: 1px solid #E6ECF5; border-radius: 18px; padding: 14px; }
    .ono { font-size: 12px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: #64748B; margin-bottom: 10px; }
    .items { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .item { background: #fff; border: 1px solid #EDF2F7; border-radius: 14px; padding: 8px; }
    .thumb { position: relative; height: 120px; border-radius: 10px; background: linear-gradient(135deg, #254F86, #5B7CF9); overflow: hidden; display: flex; align-items: center; justify-content: center; }
    .mono { color: rgba(255,255,255,.9); font-size: 40px; font-weight: 800; }
    .thumb img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
    .slot { position: absolute; left: 6px; top: 6px; background: rgba(255,255,255,.94); font-size: 11px; font-weight: 800; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
    .iname { font-size: 12px; font-weight: 700; margin-top: 8px; line-height: 1.3; }
    .ibrand { font-size: 11px; color: #64748B; margin-top: 2px; }
    .score { height: 4px; background: #EDF2F7; border-radius: 2px; margin-top: 6px; }
    .score i { display: block; height: 100%; background: #254F86; border-radius: 2px; }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .check { display: flex; gap: 10px; align-items: flex-start; background: #F8FAFF; border-radius: 12px; padding: 10px 12px; font-size: 13px; }
    .check b { display: block; }
    .check span span { color: #64748B; }
    .pill { font-size: 11px; font-weight: 800; padding: 3px 10px; border-radius: 999px; flex-shrink: 0; margin-top: 1px; }
    .pass .pill { background: #E9FBF4; color: #0E9F6E; }
    .fail .pill { background: #FDECEC; color: #E02424; }
    footer { padding: 0 40px 40px; max-width: 1240px; margin: 0 auto; color: #64748B; font-size: 13px; }
    footer code { background: #fff; padding: 2px 8px; border-radius: 6px; border: 1px solid #E2E8F0; }
  </style></head><body>
  <header><h1>AI Stylist — multi-account e2e proof</h1>
    <p>4 accounts · 2 modes (wardrobe / new clothing) · real catalog + real pipeline modules · no network · no AI spend</p>
    <p>Pipeline: <code>slot mapping → rankItemsForStyle → sanitize → deckDedupe gate</code> · ${esc(stamp)}</p>
    <span class="score-top">${passed}/${total} checks green</span></header>
  <main>${results.map(personaSection).join('')}</main>
  <footer><p>Why this proves correctness: each account section shows the <b>actual items the pipeline picked</b> (names, slots, style scores) plus the <b>same assertions the jest suite runs</b> — completeness, pairwise uniqueness, source purity (wardrobe vs Supabase catalog), and style discipline. The report script and the tests share one harness, so a green screenshot cannot drift from a green test run.</p></footer>
  </body></html>`;

  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/report.html`, html);
  console.log(`report: ${OUT}/report.html`);

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(`file://${OUT}/report.html`);
  await page.waitForTimeout(2500); // let CDN images settle
  await page.screenshot({ path: `${OUT}/report.png`, fullPage: true });
  console.log(`screenshot: ${OUT}/report.png`);
  for (let k = 0; k < results.length; k++) {
    const el = page.locator(`#persona-${k}`);
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(400);
    await el.screenshot({ path: `${OUT}/persona-${k}.png` });
    console.log(`screenshot: ${OUT}/persona-${k}.png`);
  }
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
