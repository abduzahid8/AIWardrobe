import test from "node:test";
import assert from "node:assert/strict";
import {acceptedReferences} from "./reference_library.mjs";

const sha="a".repeat(64);
const base={
  schema_version:1,source:"instagram_creator_export",account:"sainttnoirclub",
  approved_for_training:0,
  assets:[{source_account:"sainttnoirclub",sha256:sha,local_path:"images/"+sha+".jpg",
    review_status:"pending_creator_ownership_and_outfit_review",
    rights_confirmed:false,contains_full_outfit:null,
    style_tags:[],occasion_tags:[],quality_score:null}],
};

test("imported but unreviewed photos cannot be selected for fashion training",()=>{
  assert.deepEqual(acceptedReferences(base,{account:"sainttnoirclub"}),[]);
});

test("only owner-confirmed, full, high-quality curated reference is eligible",()=>{
  const approved={...base,assets:[{...base.assets[0],review_status:"approved_for_reference_use",
    rights_confirmed:true,contains_full_outfit:true,
    style_tags:["Old_Money".toLowerCase(),"classic"],occasion_tags:["work"],quality_score:0.91}]};
  const output=acceptedReferences(approved,{account:"sainttnoirclub",style:"old_money",occasion:"work"});
  assert.equal(output.length,1);
  assert.equal(output[0].sha256,sha);
  assert.equal(output[0].quality,0.91);
  assert.equal(acceptedReferences(approved,{style:"sport",occasion:"work"}).length,0);
});

test("account mismatch blocks imports; edited paths cannot escape private library",()=>{
  assert.throws(()=>acceptedReferences(base,{account:"different"}),/account_mismatch/);
  const altered={...base,assets:[{...base.assets[0],review_status:"approved_for_reference_use",
    rights_confirmed:true,contains_full_outfit:true,style_tags:["classic"],
    occasion_tags:["work"],quality_score:0.9,local_path:"../../secret.jpg"}]};
  assert.deepEqual(acceptedReferences(altered),[]);
});

test("self-reported permission does not approve reposts or missing ownership",()=>{
  const altered={...base,assets:[{...base.assets[0],review_status:"approved_for_reference_use",
    rights_confirmed:false,contains_full_outfit:true,style_tags:["classic"],
    occasion_tags:["work"],quality_score:0.9}]};
  assert.equal(acceptedReferences(altered).length,0);
});
