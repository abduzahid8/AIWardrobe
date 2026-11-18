import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EDITORIAL_TEXT_MOODBOARDS_VERSION,editorialTextMoodboards,
} from './editorial_text_moodboards.mjs';
const slots=['top','outerwear','bottom','shoes'];

test('real 46 user-caption references offer distinct Fashion EDITORIAL directions for Classic Work',()=>{
  const a=editorialTextMoodboards({style:'classic',occasion:'Work'});
  assert.match(EDITORIAL_TEXT_MOODBOARDS_VERSION,/text-architecture/);
  assert.equal(a.length,3);
  assert.ok(a.every(x=>slots.every(slot=>
    typeof x.roleDescriptions[slot]==='string'&&x.roleDescriptions[slot].length>5)));
  const lastLayers=a.map(x=>x.roleDescriptions.outerwear.split(/[\s-]+/).at(-1));
  assert.equal(new Set(lastLayers).size,3,
    'inspiration must not consist of three copies of the SAME jacket/cardigan architecture');
  const json=JSON.stringify(a);
  assert.doesNotMatch(json,/photo_[0-9]|images\/|media_sha256|instagram|creator|filename|rights_status|https?:\/\//);
  assert.deepEqual(a,editorialTextMoodboards({style:'classic',occasion:'Work'}));
});

test('real Date-night inspiration explores alternative silhouette/material pairs, not three cardigans',()=>{
  const c=editorialTextMoodboards({style:'classic',occasion:'Date night'});
  assert.equal(c.length,3);
  assert.equal(new Set(c.map(x=>x.roleDescriptions.outerwear
    .split(/[\s-]+/).at(-1))).size,3);
  assert.ok(c.some(x=>/jacket|overcoat|blouson|blazer/.test(x.roleDescriptions.outerwear)),
    'at least one real date image caption should inspire something beyond another cardigan');
  assert.ok(c.every(x=>x.paletteIdeas.length>0));
});

test('personal lookbook inspiration balances actual user-captioned shirt/tee vs knit compositions BEFORE AI designing',()=>{
  const work=editorialTextMoodboards({style:'classic',occasion:'Work'});
  const date=editorialTextMoodboards({style:'classic',occasion:'Date night'});
  assert.equal(work.length,3);
  assert.equal(date.length,3);
  assert.ok(work.some(row=>/\b(?:collared|dress|button)\s*-?\s*shirt\b|\bdress shirt\b/i
    .test(row.roleDescriptions.top)),
    'do not force the AI artist to see three high-neck knit office references when real human-shirt references exist');
  assert.ok(date.some(row=>/\b(?:t[ -]?shirt|tee|polo)\b/i.test(
    row.roleDescriptions.top)),
    'date fashion designer should see a genuine tee/polo owner concept as well as high-neck looks');
  for(const deck of [work,date]){
    assert.ok(deck.some(x=>/\b(?:knit|turtleneck|roll-neck|mockneck)\b/i.test(
      x.roleDescriptions.top)),
      'keep classic cold-weather knit art directions; this is not a keyword item ban');
    assert.ok(new Set(deck.map(x=>x.roleDescriptions.outerwear.toLowerCase()
      .split(/[\s-]+/).at(-1))).size>=3,
      'create genuinely different outer-layer construction reference silhouettes');
    assert.ok(deck.every(row=>
      slots.every(slot=>typeof row.roleDescriptions[slot]==='string'&&
        row.roleDescriptions[slot].length>=6)),
      'all inspiration must originate in human-captioned complete real ideas');
  }
  const dry=JSON.stringify([...work,...date]);
  assert.doesNotMatch(dry,/photo_[0-9]+|media_sha256|https?:\/\/|user_id|rights_basis/,
    'no missing photo rights or merchant links bypassed by reference retrieval');
});

test('unmatched activities and explicit user visual exclusions never get invented style references',()=>{
  assert.deepEqual(editorialTextMoodboards({style:'classic',occasion:'Workout'}),[]);
  assert.deepEqual(editorialTextMoodboards({style:'unknown',occasion:'Date night'}),[]);
  assert.deepEqual(editorialTextMoodboards({style:'classic'}),[]);
  const c=editorialTextMoodboards({style:'classic',occasion:'Date night',
    preferences:{avoidColors:['brown','navy']}});
  assert.ok(c.every(x=>!x.paletteIdeas.some(v=>['brown','navy'].includes(v))));
  assert.ok(c.every(x=>!slots.some(slot=>x.roleDescriptions[slot]
    .split(/[^a-z]+/i).some(v=>['brown','navy'].includes(v.toLowerCase())))));
});
