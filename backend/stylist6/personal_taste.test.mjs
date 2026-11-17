import test from 'node:test';
import assert from 'node:assert/strict';
import { personalTaste, evaluatePersonalItem, evaluatePersonalLook } from './personal_taste.mjs';

const item = (extra = {}) => ({
  id: 'sku-navy-1', name: 'Navy wool blazer', brand: 'Brand House',
  color: 'navy', ...extra,
});
const descriptor = (extra = {}) => ({
  color: 'navy', pattern: 'stripe', volume: 'slim', ...extra,
});

test('an unconfigured or default-balanced profile does not bias a look', () => {
  for (const input of [null, undefined, {}, {fitPreference:'balanced'}, 'bad', []]) {
    const taste = personalTaste(input);
    assert.equal(taste.active, false);
    assert.deepEqual(evaluatePersonalItem(item(), descriptor(), taste), {
      blocked: null, score: 0, matches: [],
    });
    assert.deepEqual(evaluatePersonalLook([{ item: item(), descriptor: descriptor() }], taste),
      { adjustment: 0, matches: [], matchedItems: 0 });
  }
});

test('explicit navy, brand, striped pattern and fitted silhouette help the same real item', () => {
  const taste = personalTaste({
    favoriteColors: [' NAVY ', 'navy'],
    favoriteBrands: ['BRAND HOUSE'],
    likedPatterns: ['stripes'],
    fitPreference: 'fitted',
  });
  const scored = evaluatePersonalItem(item(), descriptor(), taste);
  assert.equal(scored.blocked, null);
  assert.equal(scored.score, 0.12); // capped: no single item may dominate coherence
  assert.deepEqual(scored.matches.map(x => x.type),
    ['favorite_color', 'favorite_brand', 'liked_pattern', 'preferred_fit']);
  assert.ok(scored.matches.every(x=>x.productId==='sku-navy-1'));
  const look = evaluatePersonalLook([
    {item:item(), descriptor:descriptor()},
    {item:item({id:'sku-bottom',brand:'Other',color:'grey'}), descriptor:descriptor({color:'grey',pattern:'solid',volume:'regular'})},
    {item:item({id:'sku-shoes',brand:'Other',color:'brown'}), descriptor:descriptor({color:'brown',pattern:'solid',volume:'unknown'})},
  ], taste);
  assert.ok(look.adjustment > 0 && look.adjustment <= .085);
  assert.equal(look.matchedItems,1);
  assert.ok(look.matches.every(x=>x.productId==='sku-navy-1'));
});

test('avoided brands are exact, case-insensitive exclusions and not fuzzy substrings', () => {
  const taste = personalTaste({avoidBrands:['  BRAND HOUSE  ']});
  assert.equal(evaluatePersonalItem(item(), descriptor(),taste).blocked,'excluded_brand');
  assert.equal(evaluatePersonalItem(item({brand:'Brand House Studio'}),descriptor(),taste).blocked,null);
  assert.equal(evaluatePersonalItem(item({brand:'Other'}),descriptor(),taste).blocked,null);
  assert.equal(evaluatePersonalItem(item({brand:null}),descriptor(),taste).blocked,null);
});

test('fit dislike has bounded soft weight and unknown evidence is neutral', () => {
  const taste=personalTaste({fitPreference:'loose',favoriteColors:['navy'],favoriteBrands:['Brand House']});
  const mismatch = evaluatePersonalItem(item({brand:'Different',color:'grey'}),
    descriptor({color:'grey',volume:'slim'}),taste);
  assert.equal(mismatch.score,-0.025);
  const unknown = evaluatePersonalItem(item({brand:'',color:'unknown'}),
    descriptor({color:'unknown',volume:'unknown',pattern:'unknown'}),taste);
  assert.equal(unknown.score,0);
  const matching = evaluatePersonalItem(item({brand:'Different',color:'grey'}),
    descriptor({color:'grey',volume:'wide'}),taste);
  assert.equal(matching.score,.025);
});

test('hostile and oversized preferences are bounded without guessing body shape or price', () => {
  const taste=personalTaste({
    favoriteColors: Array.from({length:100},(_,i)=>'color'+i),
    avoidBrands:[{},null,44,'Brand House'],
    fitPreference:'tall',
    bodyType:'athletic',priceRange:{min:1,max:1,currency:'USD'},
  });
  assert.ok(taste.favoriteColors.size<=16);
  assert.equal(taste.desiredFit,null);
  assert.equal(evaluatePersonalItem(item(),descriptor(),taste).blocked,'excluded_brand');
  assert.equal(evaluatePersonalLook([],taste).adjustment,0);
});
