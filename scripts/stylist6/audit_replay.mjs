// Audit the frozen fashion replay for concrete styling failures.
// Usage: node scripts/stylist6/audit_replay.mjs .private/fashion-replay-before.json
import fs from 'node:fs';

const report = JSON.parse(fs.readFileSync(process.argv[2] || '.private/fashion-replay-before.json', 'utf8'));
const catalog = report.catalog;
const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
const arms = ['stylist5', 'stylist6Previous', 'stylist6Fashion'];

const deckScore = (looks) => looks.reduce((sum, look) => sum + (look.coherence ?? 0), 0);
const paired = { s5Higher: [], p6Higher: [], f6HigherVsS5: [], f6HigherVsP6: [] };
for (const row of report.results) {
  const s5 = deckScore(row.arms.stylist5), p6 = deckScore(row.arms.stylist6Previous), f6 = deckScore(row.arms.stylist6Fashion);
  if (row.arms.stylist5.length && row.arms.stylist6Fashion.length) (s5 > f6 ? paired.s5Higher : paired.f6HigherVsS5).push(row.scenario.id);
  if (row.arms.stylist6Previous.length && row.arms.stylist6Fashion.length) (p6 > f6 ? paired.p6Higher : paired.f6HigherVsP6).push(row.scenario.id);
}
console.log('deck-level S5 higher:', paired.s5Higher.join(', ') || 'none');
console.log('deck-level prev6 higher:', paired.p6Higher.join(', ') || 'none');

// Warm-context outerwear + footwear audit across every arm.
const flagRows = [];
for (const row of report.results) {
  const temp = row.scenario.weather.temp;
  for (const arm of arms) {
    row.arms[arm].forEach((look, index) => {
      const items = look.ids.map((id) => catalog[id]);
      const names = items.map((p) => `${p.slot}:${p.name}`).join(' | ');
      const flags = [];
      if (temp >= 25 && look.layered) flags.push('warm_outerwear');
      const shoes = items.find((p) => p.slot === 'shoes');
      if (shoes && /waterproof|hiking|trail|wellington|snow|insulated/i.test(shoes.name) && temp >= 20) flags.push('warm_rugged_footwear');
      if (shoes && temp >= 25 && /boot/i.test(shoes.name)) flags.push('warm_boots');
      const outer = items.find((p) => p.slot === 'outerwear');
      if (outer && temp >= 25 && /parka|puffer|quilted|insulated|down|wool coat|overcoat|trench/i.test(outer.name)) flags.push('warm_heavy_outerwear');
      if (!look.coherent) flags.push('incoherent');
      if (flags.length) flagRows.push({ id: row.scenario.id, arm, look: index + 1, temp, occasion: row.scenario.occasion, flags, names });
    });
  }
}
console.log('\nflagged looks:', flagRows.length);
for (const f of flagRows) console.log(` ${f.id} ${f.arm} L${f.look} ${f.temp}°C [${f.flags.join(',')}] ${f.names}`);

// Deck repetition: mean styling distance and repeated SKUs inside each deck.
console.log('\ndeck diversity (styling distance / distinct SKUs):');
for (const row of report.results) {
  const parts = arms.map((arm) => {
    const looks = row.arms[arm];
    const skus = new Set(looks.flatMap((l) => l.ids)).size;
    return `${arm.slice(0, 3)}:${skus}`;
  });
  if (parts.some((p) => p.endsWith(':0'))) continue;
}
console.log('\nmean coherence by arm and temperature band:');
const bands = [[0, 15], [15, 22], [22, 30]];
for (const [lo, hi] of bands) {
  const rows = report.results.filter((r) => r.scenario.weather.temp >= lo && r.scenario.weather.temp < hi);
  const line = arms.map((arm) => {
    const looks = rows.flatMap((r) => r.arms[arm]);
    return `${arm}=${avg(looks.map((l) => l.coherence))?.toFixed(3)} (n=${looks.length})`;
  });
  console.log(` ${lo}-${hi}°C: ${line.join('  ')}`);
}
