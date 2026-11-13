/**
 * Evidence-driven Stylist 6 release gate. This never contacts vendors, reads
 * user accounts or deploys. It validates an owner-reviewed, private evidence
 * manifest with independent fashion votes and merchant/hosted receipts.
 * Passing this gate is NOT authorization to change production.
 */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const SHA = /^[a-f0-9]{40}$/;
const DIGEST = /^[a-f0-9]{64}$/;
const whole = x => Number.isSafeInteger(x) && x >= 0;
const yes = x => x === true;

export function evaluateStylist6Release(manifest) {
  const failures = [];
  const gate = (id, ok, reason) => { if (!ok) failures.push({ gate: id, reason }); };
  const m = manifest && typeof manifest === 'object' && !Array.isArray(manifest) ? manifest : {};
  gate('schema', m.schema === 'aiw.stylist6.release.v1', 'Evidence schema absent or invalid');

  const build = m.build || {};
  gate('immutable-build', SHA.test(build.gitSha || '') && SHA.test(build.stagingSha || '') &&
    build.gitSha === build.stagingSha && yes(build.productionUntouched) && yes(build.rollbackTested),
    'Tested staging SHA must equal candidate SHA; production unchanged and rollback verified');
  const ci = m.ci || {};
  gate('ci-security', ci.gitSha === build.gitSha &&
    ['typescript','jest','node','build','rls','migrations','security'].every(k => yes(ci[k])),
    'Security, DB, TypeScript and test gates must be green on the exact SHA');

  const hard = m.hardConstraints || {};
  gate('hard-constraints', whole(hard.auditedLooks) && hard.auditedLooks >= 100 &&
    hard.violations === 0 &&
    ['catalog','wardrobe','mixed','lockedAnchor'].every(k => yes(hard.modes?.[k])),
    '100+ audited looks: zero hard violations, all sourcing modes and anchored items');
  const diversity = m.diversity || {};
  gate('three-real-directions',
    whole(diversity.catalogFeasibleScenarios) && diversity.catalogFeasibleScenarios >= 30 &&
    whole(diversity.distinctThreeDelivered) &&
    diversity.distinctThreeDelivered <= diversity.catalogFeasibleScenarios &&
    diversity.distinctThreeDelivered / diversity.catalogFeasibleScenarios >= .90,
    'Need three genuinely distinct directions in >=90% of 30+ feasible requests');

  const visual = m.visual || {};
  gate('visual-truth', whole(visual.fullyAdvertisedDecks) && visual.fullyAdvertisedDecks >= 30 &&
    visual.allThreeApproved === visual.fullyAdvertisedDecks &&
    visual.unreviewedPresentedAsVerified === 0 &&
    whole(visual.maxCallsPerRequest) && visual.maxCallsPerRequest <= 4 &&
    yes(visual.mediaRightsReviewed),
    '30+ complete visual decks: 3/3 actually approved and media-rights reviewed');

  const blind = m.humanBlind || {};
  const rows = Array.isArray(blind.scenarios) ? blind.scenarios : [];
  const ids = new Set();
  let wins = 0, losses = 0, ties = 0, structurallyValid = true;
  for (const row of rows) {
    if (!row || typeof row.id !== 'string' || !row.id || ids.has(row.id) ||
      !DIGEST.test(row.auditSha256 || '') || !Array.isArray(row.votes) ||
      row.votes.length < 3) { structurallyValid = false; continue; }
    ids.add(row.id);
    const reviewers = new Set();
    let s6 = 0, s5 = 0, validVotes = true;
    for (const vote of row.votes) {
      if (!vote || typeof vote.reviewer !== 'string' || !vote.reviewer ||
        reviewers.has(vote.reviewer) ||
        !['s6','s5','tie'].includes(vote.choice)) { validVotes = false; break; }
      reviewers.add(vote.reviewer);
      if (vote.choice === 's6') s6++;
      else if (vote.choice === 's5') s5++;
    }
    if (!validVotes) { structurallyValid = false; continue; }
    if (s6 > row.votes.length / 2) wins++;
    else if (s5 > row.votes.length / 2) losses++;
    else ties++;
  }
  gate('human-blind', structurallyValid && ids.size === rows.length,
    'Every held-out case needs 3+ distinct blinded reviewer votes and an audit digest');
  gate('independent-fashion', rows.length >= 30 && ids.size === rows.length &&
    yes(blind.independentOfStylistModel) && wins / rows.length >= .70,
    'Need 30+ independent paired cases, >=70% S6 majority wins; ties count as non-wins');

  const merchant = m.merchant || {};
  gate('uzbekistan-purchase', merchant.country === 'UZ' &&
    whole(merchant.advertisedItems) && merchant.advertisedItems >= 100 &&
    merchant.verifiedOffers === merchant.advertisedItems &&
    whole(merchant.eligibleDistinctShoes) && merchant.eligibleDistinctShoes >= 200 &&
    merchant.missingSizeOrShipping === 0 &&
    yes(merchant.exactVariantAndPrice) && yes(merchant.imagesDisplayLicensed) &&
    yes(merchant.imagesInferenceLicensed) && DIGEST.test(merchant.merchantAuditSha256 || ''),
    '100% seller-verified size/stock/delivery/offers, >=200 eligible shoes, image rights and audit');

  const hosted = m.hosted || {};
  const requests = Array.isArray(hosted.requests) ? hosted.requests : [];
  const samples = requests.map(x => x?.wallMs).filter(x => Number.isFinite(x) && x >= 0);
  const sorted = [...samples].sort((a,b) => a-b);
  const p95 = sorted.length ? sorted[Math.ceil(sorted.length*.95)-1] : null;
  gate('hosted-performance', requests.length >= 30 && samples.length === requests.length &&
    p95 !== null && p95 <= 25000 &&
    requests.every(x => whole(x.providerCalls) && x.providerCalls <= 4 && x.debitCount === 1) &&
    yes(hosted.replayIsIdempotent) && yes(hosted.jwtHmacRlsVerified) &&
    yes(hosted.spendCapVerified) && hosted.duplicateDebits === 0,
    '30+ hosted calls p95 <=25s, <=4 provider calls, single debit and safe replay/security/spend');

  return {
    verdict: failures.length ? 'NO_GO' : 'ENGINEERING_RELEASE_CANDIDATE',
    deploymentAuthorized: false,
    failures,
    measured: {
      humanComparable: rows.length,
      s6Wins: wins, s5Wins: losses, ties,
      winRate: rows.length ? wins / rows.length : null,
      hostedSamples: samples.length, p95Ms: p95,
      verifiedOffers: whole(merchant.verifiedOffers) ? merchant.verifiedOffers : null,
      eligibleShoes: whole(merchant.eligibleDistinctShoes) ? merchant.eligibleDistinctShoes : null,
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  let input;
  try {
    if (!process.argv[2] || process.argv[2].startsWith('--')) throw new Error('evidence path required');
    input = JSON.parse(await readFile(process.argv[2], 'utf8'));
  } catch {
    console.error('NO_GO: supply a valid, private owner-reviewed evidence manifest.');
    process.exitCode = 2;
  }
  if (input) {
    const result = evaluateStylist6Release(input);
    console.log(JSON.stringify(result, null, 2));
    if (result.verdict !== 'ENGINEERING_RELEASE_CANDIDATE') process.exitCode = 1;
  }
}
