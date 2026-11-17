/** Offline named-fixture audit. No images, subjective ratings, network or user data. */
import {writeFileSync} from 'node:fs';
import {assessOutfitQuality} from '../../src/domain/occasionRules';
import {describeGarment} from '../../src/domain/garmentDescriptor';
import {garmentContextReason} from '../../src/domain/garmentContext';
import {assembleDiverseDeck, productIdentityKeys} from '../../src/domain/outfitDiversity';
import {stylingContext} from '../../src/domain/stylingContext';
import {fixtureLook, stylingV193Fixtures} from '../../__tests__/fixtures/stylingV193';

const started = performance.now();
const reports = stylingV193Fixtures.map(fixture => {
  const start = performance.now(), items = fixtureLook(fixture.names, fixture.id);
  const context = stylingContext({style: fixture.style || 'semi_classic', occasion: fixture.occasion || 'Date night'});
  const quality = assessOutfitQuality(items, context.style, context.occasion, context.weather);
  return {id: fixture.id, items, expectedAcceptance: fixture.accepted,
    accepted: quality.acceptance.passed, rejection: quality.acceptance.reason,
    eligiblePoolSize: items.filter(item => !garmentContextReason(item, context)).length,
    signals: quality.evidence, reasons: quality.reasons, unknownContext: context.uncertainty,
    descriptors: items.map(describeGarment), score: quality.score, durationMs: performance.now() - start};
});
const approved = reports.filter(r => r.accepted && !stylingV193Fixtures.find(f => f.id === r.id)!.occasion);
const selected = assembleDiverseDeck({candidates: approved, itemsOf: c => c.items,
  qualityOf: c => c.score, usedKeys: new Set<string>(), target: 3, maxSearchSteps: 512});
const keys = selected.deck.flatMap(c => c.items.flatMap(productIdentityKeys));
const result = {scope: 'deterministic named garment fixtures; no visual human score or live inference',
  reports, deck: {candidateCount: approved.length, selected: selected.deck.map(c => c.id),
    selectedCount: selected.deck.length, duplicateIdentityKeyCount: keys.length - new Set(keys).size,
    productDiversity: selected.productDiversity}, durationMs: performance.now() - started};
const output = process.argv[2];
if (output) writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({fixtures: reports.length, accepted: reports.filter(r => r.accepted).length,
  expectationMismatches: reports.filter(r => r.accepted !== r.expectedAcceptance).map(r => r.id),
  deck: result.deck, durationMs: result.durationMs}));
