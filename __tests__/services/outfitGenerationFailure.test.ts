/** @jest-environment node */
import { createClient } from '@supabase/supabase-js';
import { outfitGenerationFailure } from '../../src/services/outfitGenerationFailure';

test.each([
 ['no_outfit_within_budget','outfitMaker.budgetTooLow'],
 ['no_items_within_budget','outfitMaker.noItemsWithinBudget'],
 ['no_matching_clothing','outfitMaker.noMatchingClothing'],
 ['invalid_outfit_budget','outfitMaker.invalidBudget'],
])('maps known server code %s to localized message %s',async(code,key)=>{
 expect((await outfitGenerationFailure({code})).translationKey).toBe(key);
});
test('invalid-budget error field is recognized without displaying server text',async()=>{
 expect((await outfitGenerationFailure({error:'invalid_outfit_budget'})).translationKey).toBe('outfitMaker.invalidBudget');
});
test('decodes the installed Supabase HTTP error Response and leaves its body readable',async()=>{
 const client=createClient('https://outfit-errors.invalid','fixture-anon',{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:async()=>new Response(JSON.stringify({success:false,code:'no_outfit_within_budget'}),{status:422,headers:{'Content-Type':'application/json'}})}});
 const {data,error}=await client.functions.invoke('generate-outfits',{body:{}});
 expect(error).toBeTruthy();
 expect((await outfitGenerationFailure(data,error)).translationKey).toBe('outfitMaker.budgetTooLow');
 expect(await error.context.json()).toEqual({success:false,code:'no_outfit_within_budget'});
});
test.each([{error:'private provider diagnostic'}, {code:'__proto__'},null])('unknown payload %j uses generic local copy',async data=>{
 const failure=await outfitGenerationFailure(data);
 expect(failure.translationKey).toBe('outfitMaker.generateFailed');expect(failure.message).toBe('outfit_generation_failed');
});
test('malformed HTTP response body uses generic local copy',async()=>{
 expect((await outfitGenerationFailure(null,{context:new Response('not JSON')})).translationKey).toBe('outfitMaker.generateFailed');
});
test('a hung error-body read cannot leave the outfit screen waiting',async()=>{
 jest.useFakeTimers();
 try{
  const pending=outfitGenerationFailure(null,{context:{clone:()=>({json:()=>new Promise(()=>{})})}});
  await jest.advanceTimersByTimeAsync(1500);
  expect((await pending).translationKey).toBe('outfitMaker.generateFailed');expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});
test.each(['en','ru','uz'])('%s includes every mapped outfit failure message',locale=>{
 const translations=require(`../../i18n/locales/${locale}.json`);
 for(const key of ['budgetTooLow','noItemsWithinBudget','noMatchingClothing','invalidBudget','generateFailed','aiUnavailable']) expect(translations.outfitMaker[key].length).toBeGreaterThan(20);
});

test('unknown prices have their own localized failure rather than a budget-too-low claim', async () => {
    expect((await outfitGenerationFailure({code:'unverified_shop_pricing'})).translationKey).toBe('outfitMaker.shopPricesUnavailable');
});

it.each([
 [400,'outfitMaker.invalidRequest'],[413,'outfitMaker.requestTooLarge'],[408,'outfitMaker.requestUploadTimedOut'],
])('request admission HTTP %s has actionable localized guidance',async(status,key)=>{
 const context=new Response(JSON.stringify({error:'invalid_generation_request'}),{status});
 expect((await outfitGenerationFailure(null,{context})).translationKey).toBe(key);
 expect((await context.json()).error).toBe('invalid_generation_request');
});
it.each(['en','ru','uz'])('%s has request-limit, size and upload guidance',locale=>{
 const copy=require(`../../i18n/locales/${locale}.json`).outfitMaker;
 for(const key of ['invalidRequest','requestTooLarge','requestUploadTimedOut'])expect(copy[key]?.length).toBeGreaterThan(20);
});

test('an AI response failure does not tell the user to add wardrobe items',async()=>{
 expect((await outfitGenerationFailure({error:'ai_generation_unavailable'})).translationKey).toBe('outfitMaker.aiUnavailable');
});
test('the actual SDK503 context preserves the AI-specific failure message',async()=>{
 const context=new Response(JSON.stringify({error:'ai_generation_unavailable',outfits:[]}),{status:503});
 expect((await outfitGenerationFailure(null,{context})).translationKey).toBe('outfitMaker.aiUnavailable');
});

test('unknown failure diagnostics retain only a bounded category and status',async()=>{
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});
 try {
  const privateValue='private-token-and-user-input';
  const failure=await outfitGenerationFailure(null,{name:privateValue,message:privateValue,headers:{Authorization:privateValue},context:new Response(JSON.stringify({error:privateValue}),{status:500})});
  expect(failure.translationKey).toBe('outfitMaker.generateFailed');
  expect(warn).toHaveBeenCalledWith('[stylist-request-failure]',{phase:'http_body',sdkError:'unknown',httpStatus:500});
  expect(JSON.stringify(warn.mock.calls)).not.toContain(privateValue);
 } finally {warn.mockRestore()}
});

 test.each(['FunctionsFetchError', 'FunctionsRelayError'])('%s gives connection guidance', async name => {
  expect((await outfitGenerationFailure(null, {name})).translationKey).toBe('outfitMaker.connectionFailed');
 });
