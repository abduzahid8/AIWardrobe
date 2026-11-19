import test from 'node:test';
import assert from 'node:assert/strict';
import {
  photoRewardContextHash,photoRewardContextStatus,
  anonymousPhotoJournalAllowed,
} from './photo_reward_journal.mjs';
import {photoHistoryContextHash} from './photo_reward_history.mjs';

const SECRET='server-only-photo-context-secret-for-safe-test';

test('same actual app UI style maps to same private journal and reader context',()=>{
  const prefs={
    fitPreference:'loose',avoidColors:['navy'],
    favoriteColors:['cream'],favoriteBrands:['Massimo Dutti'],
    likedPatterns:[],styleGoals:['shop smarter'],
  };
  const real={style:'Old Money',occasion:'Date night',preferences:prefs};
  const canonical={style:'old_money',occasion:'date',
    preferences:{...prefs,favoriteColors:['cream'],avoidColors:['navy']}};
  assert.equal(photoRewardContextStatus(real),'supported');
  assert.equal(photoRewardContextHash(real,SECRET),
    photoRewardContextHash(canonical,SECRET));
  assert.equal(photoRewardContextHash(real,SECRET),
    photoHistoryContextHash(real,SECRET));
  assert.match(photoRewardContextHash(real,SECRET),/^[a-f0-9]{64}$/);
  assert.equal(anonymousPhotoJournalAllowed({
    sourceMode:'catalog',requiredIds:[],preferences:prefs,
  }),true);
});

test('unrecognized user style follows the exact engine casual fallback, not a garment rule',()=>{
  const a={style:'Designer custom style',occasion:'Work',
    preferences:{fitPreference:'balanced'}};
  const b={style:'casual',occasion:'work',
    preferences:{fitPreference:'balanced'}};
  assert.equal(photoRewardContextStatus(a),'supported');
  assert.equal(photoRewardContextHash(a,SECRET),
    photoHistoryContextHash(b,SECRET));
});

test('unrelated personal occasions, unsafe identity metadata and malformed preferences abstain',()=>{
  const bare={style:'classic',occasion:'work',preferences:{}};
  for(const [context,reason] of [
    [{...bare,style:''},'unsupported_style'],
    [{...bare,occasion:'Private personal wedding'},'unsupported_occasion'],
    [{...bare,preferences:{ownerEmail:'private@example.com'}},
      'unsupported_preferences'],
    [{...bare,preferences:{fitPreference:'secret-private-body-type'}},
      'unsupported_preferences'],
    [{...bare,preferences:{favoriteColors:['mailto:private@example.com']}},
      'unsupported_preferences'],
  ]){
    assert.equal(photoRewardContextStatus(context),reason);
    assert.equal(photoRewardContextHash(context,SECRET),null);
  }
});

test('genuine photo context hashes cannot be used without the server key',()=>{
  const context={style:'Classic',occasion:'Work',preferences:{
    favoriteColors:['navy'],fitPreference:'fitted',
  }};
  assert.equal(photoRewardContextHash(context,'weak'),null);
  assert.notEqual(photoRewardContextHash(context,SECRET),
    photoRewardContextHash(context,'different-server-photo-context-secret'));
});
