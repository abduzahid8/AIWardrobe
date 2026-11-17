// Offline audit of actual review decks against separately obtained evidence.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { outfitPurchaseEvidence, trialImageDecision } from '../../backend/stylist6/retail_evidence.mjs';

const folder = fs.realpathSync(process.argv[2]);
function read(name, fallback) {
  try { return JSON.parse(fs.readFileSync(path.join(folder, name), 'utf8')); }
  catch (error) { if (error.code === 'ENOENT' && fallback !== undefined) return fallback; throw error; }
}
const rights = read('image-rights.json', null);
const offers = read('merchant-offers.json', []);
const requirements = read('purchase-requirements.json', {});
const hashes = read('references/hashes.json');
const scenarios = new Map(read('scenarios.json').map((scenario) => [scenario.id, scenario]));
const replay = read('replay.json');
const hashFile = (name) => createHash('sha256').update(fs.readFileSync(path.join(folder, name))).digest('hex');
if (hashFile('inventory.json') !== replay.inventorySha256 || hashFile('scenarios.json') !== replay.scenariosSha256) {
  throw new Error('retail_audit_fixture_mismatch');
}
const slots = new Map(replay.admitted.map(({ id, macroCategory }) => [id, macroCategory]));
const inventory = new Map(read('inventory.json').map((entry) => [entry.row.id, entry]));
const reviewed = new Set();
const verifiedReferences = [];
let totalLooks = 0, verifiedLooks = 0;
for (const row of replay.results) {
  const scenario = scenarios.get(row.id);
  const reference = scenario.reference;
  if (trialImageDecision(rights, { kind: 'reference', id: reference, sha256: hashes[reference] }) === 'cleared' &&
      hashFile('references/' + reference + '.jpg') === hashes[reference]) {
    verifiedReferences.push(reference);
  }
  for (const look of [...row.baseline, ...row.v6Retrieval]) {
    const items = look.map((sku) => {
      reviewed.add(sku);
      return { ...inventory.get(sku).row, macroCategory: slots.get(sku) };
    });
    totalLooks++;
    const requirement = requirements[row.id];
    if (requirement) {
      const budget = scenario.budget;
      if (budget && requirement.currency !== budget.currency) continue;
      const result = outfitPurchaseEvidence(items, offers,
        { ...requirement, budget: budget?.amount ?? requirement.budget });
      if (result.status === 'verified') verifiedLooks++;
    }
  }
}
const clearedProducts = [...reviewed].filter((id) => {
  const record = inventory.get(id);
  return trialImageDecision(rights, { kind: 'product', id, sourceUrl: record.row.image_url,
    sha256: record.photoSha256 }) === 'cleared' && hashFile(record.photo) === record.photoSha256;
});
console.log(JSON.stringify({
  reviewLooks: totalLooks, reviewUniqueSkus: reviewed.size, verifiedPurchaseLooks: verifiedLooks,
  offerEvidenceRows: offers.length, scenariosWithPurchaseRequirements: Object.keys(requirements).length,
  clearedProductImages: clearedProducts.length, clearedReferenceImages: new Set(verifiedReferences).size,
  v5Ranked: replay.results.filter(({ v5Mode }) => v5Mode === 'hosted_ranked').length,
  v6Hosted: replay.results.filter(({ hostedStatus }) => hostedStatus === 'JUDGED').length,
  independentRatings: null,
}, null, 2));
