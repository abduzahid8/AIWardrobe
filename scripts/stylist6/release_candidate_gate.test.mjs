import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateStylist6Release } from './release_candidate_gate.mjs';

const sha = 'a'.repeat(40);
const digest = 'b'.repeat(64);
const votes = ['s6','s6','s5'].map((choice, n) => ({ reviewer: 'reviewer'+n, choice }));
const scenario = i => ({ id: 'case-'+i, auditSha256: digest,
  votes: votes.map(v => ({ ...v })) });
const completeFixture = () => ({
  schema: 'aiw.stylist6.release.v1',
  build: { gitSha: sha, stagingSha: sha, productionUntouched: true, rollbackTested: true },
  ci: { gitSha: sha, typescript: true, jest: true, node: true, build: true,
    rls: true, migrations: true, security: true },
  hardConstraints: { auditedLooks: 100, violations: 0,
    modes: { catalog: true, wardrobe: true, mixed: true, lockedAnchor: true } },
  diversity: { catalogFeasibleScenarios: 30, distinctThreeDelivered: 27 },
  visual: { fullyAdvertisedDecks: 30, allThreeApproved: 30,
    unreviewedPresentedAsVerified: 0, maxCallsPerRequest: 4, mediaRightsReviewed: true },
  humanBlind: { independentOfStylistModel: true,
    scenarios: Array.from({length:30},(_,i)=>scenario(i)) },
  merchant: { country:'UZ', advertisedItems:100, verifiedOffers:100,
    eligibleDistinctShoes:200, missingSizeOrShipping:0,
    exactVariantAndPrice:true, imagesDisplayLicensed:true,
    imagesInferenceLicensed:true, merchantAuditSha256:digest },
  hosted: { requests: Array.from({length:30},()=>({wallMs:17000,providerCalls:3,debitCount:1})),
    replayIsIdempotent:true,jwtHmacRlsVerified:true,spendCapVerified:true,duplicateDebits:0 },
});

test('missing evidence is fail-closed, does not authorize production',()=>{
  const result=evaluateStylist6Release(undefined);
  assert.equal(result.verdict,'NO_GO');
  assert.equal(result.deploymentAuthorized,false);
  assert.ok(result.failures.length>=8);
});

test('complete SYNTHETIC fixture exercises the mechanical GO without ever authorizing release',()=>{
  const result=evaluateStylist6Release(completeFixture());
  assert.equal(result.verdict,'ENGINEERING_RELEASE_CANDIDATE');
  assert.equal(result.deploymentAuthorized,false);
  assert.equal(result.measured.winRate,1);
  assert.equal(result.measured.p95Ms,17000);
});

test('a win-rate below seventy percent fails even though all CI tests pass',()=>{
  const input=completeFixture();
  input.humanBlind.scenarios.slice(0,10).forEach(row=>{
    row.votes=[{reviewer:'a',choice:'s5'},{reviewer:'b',choice:'s5'},{reviewer:'c',choice:'s6'}];
  });
  const out=evaluateStylist6Release(input);
  assert.equal(out.measured.winRate,20/30);
  assert.ok(out.failures.some(x=>x.gate==='independent-fashion'));
});

test('seller inventory, sizes, rights and proof digest must all be grounded',()=>{
  const input=completeFixture();
  input.merchant.verifiedOffers=99;
  input.merchant.imagesInferenceLicensed=false;
  input.merchant.merchantAuditSha256='';
  assert.ok(evaluateStylist6Release(input).failures.some(x=>x.gate==='uzbekistan-purchase'));
});

test('missing image review, partial diversity, double debit and p95 deadline fail',()=>{
  const input=completeFixture();
  input.diversity.distinctThreeDelivered=26;
  input.visual.allThreeApproved=29;
  input.hosted.requests[29].wallMs=28000;
  input.hosted.requests[0].debitCount=2;
  const result=evaluateStylist6Release(input);
  assert.ok(['three-real-directions','visual-truth','hosted-performance'].every(g=>
    result.failures.some(x=>x.gate===g)));
});

test('duplicate scenario IDs or reviewer identities are not independent observations',()=>{
  const input=completeFixture();
  input.humanBlind.scenarios[1].id=input.humanBlind.scenarios[0].id;
  input.humanBlind.scenarios[2].votes[1].reviewer='reviewer0';
  const result=evaluateStylist6Release(input);
  assert.equal(result.verdict,'NO_GO');
  assert.ok(result.failures.some(x=>x.gate==='human-blind'));
  assert.ok(result.failures.some(x=>x.gate==='independent-fashion'));
});

test('staging code skew and missing rollback evidence block a release candidate',()=>{
  const input=completeFixture();
  input.build.stagingSha='c'.repeat(40);
  input.build.rollbackTested=false;
  assert.ok(evaluateStylist6Release(input).failures.some(x=>x.gate==='immutable-build'));
});
