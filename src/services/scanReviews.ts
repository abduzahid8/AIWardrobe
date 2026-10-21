import * as FileSystem from 'expo-file-system/legacy';
import { Platform } from 'react-native';
import { authEvents } from '../../store/authEvents';
import type { AnalysisResult } from '../features/wardrobe/useVideoAnalysis';

const ROOT = 'aiwardrobe-scan-reviews';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const id = () => 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r=Math.floor(Math.random()*16); return (c==='x'?r:(r&3)|8).toString(16); });
let work: Promise<unknown> = Promise.resolve();
const listeners = new Set<() => void>();
function serial<T>(action: () => Promise<T>): Promise<T> { const next=work.then(action); work=next.catch(()=>{}); return next; }
function notify() { for(const listener of listeners) { try { listener(); } catch { console.warn('Scan review listener failed'); } } }
function scope(owner: string) {
  const revision=authEvents.getRevision();
  const check=()=>{if(!UUID.test(owner)||owner!==authEvents.getUserId()||revision!==authEvents.getRevision())throw Error('Scan review account changed');};
  check();if(!FileSystem.documentDirectory)throw Error('Scan review storage unavailable');
  return {folder:`${FileSystem.documentDirectory}${ROOT}/${owner}/`,check};
}
interface RecordData { version:1; id:string; owner:string; createdAt:number; sourceQueueId?:string; result:AnalysisResult }
export interface ScanReviewSummary { id:string; count:number; createdAt:number }
export function subscribeScanReviews(listener:()=>void) { listeners.add(listener);return()=>{listeners.delete(listener);}; }
class InvalidScanReview extends Error {}
async function readRecord(folder:string, reviewId:string, owner:string):Promise<RecordData> {
  if(!UUID.test(reviewId))throw Error('Invalid scan review');
  const info=await FileSystem.getInfoAsync(`${folder}${reviewId}.json`);
  if(!info.exists||info.isDirectory||!Number.isSafeInteger(info.size)||info.size>32*1024*1024)throw new InvalidScanReview('Scan review unavailable');
  const text=await FileSystem.readAsStringAsync(`${folder}${reviewId}.json`);
  let value:any;
  try { value=JSON.parse(text); } catch { throw new InvalidScanReview('Invalid scan review JSON'); }
  if(!value||value.version!==1||value.id!==reviewId||value.owner!==owner||!Number.isSafeInteger(value.createdAt)||value.createdAt<=0||(value.sourceQueueId!==undefined&&(typeof value.sourceQueueId!=='string'||!/^[a-z0-9_-]{1,100}$/i.test(value.sourceQueueId)))||!Array.isArray(value.result?.detectedItems)||!value.result.detectedItems.length||value.result.detectedItems.length>20)throw new InvalidScanReview('Invalid scan review');
  const itemIds=new Set<string>();
  for(const item of value.result.detectedItems){
    if(!item||!UUID.test(item.reviewItemId)||typeof item.frameImage!=='string'||!item.frameImage.startsWith('data:image/')||typeof item.itemType!=='string')throw new InvalidScanReview('Invalid scan review item');
    if(itemIds.has(item.reviewItemId))throw new InvalidScanReview('Duplicate scan review item');
    itemIds.add(item.reviewItemId);
  }
  return value;
}
async function pendingResult(folder:string, record:RecordData):Promise<AnalysisResult> {
  const pending=[];
  for(const item of record.result.detectedItems){
    const saved=await FileSystem.getInfoAsync(`${folder}${record.id}.${item.reviewItemId}.saved`);
    if(!saved.exists || saved.isDirectory || await FileSystem.readAsStringAsync(`${folder}${record.id}.${item.reviewItemId}.saved`) !== 'saved')pending.push(item);
  }
  return {...record.result,reviewId:record.id,sourceQueueId:record.sourceQueueId,detectedItems:pending};
}
async function finishReview(folder:string, record:RecordData, check:()=>void):Promise<void> {
  check();
  if(record.sourceQueueId){
    if(!/^[a-z0-9_-]{1,100}$/i.test(record.sourceQueueId))throw Error('Invalid scan queue identity');
    await FileSystem.writeAsStringAsync(`${folder}${record.sourceQueueId}.done`,JSON.stringify({owner:record.owner,sourceQueueId:record.sourceQueueId}));check();
    const completed=JSON.parse(await FileSystem.readAsStringAsync(`${folder}${record.sourceQueueId}.done`));check();
    if(completed.owner!==record.owner||completed.sourceQueueId!==record.sourceQueueId)throw Error('Scan completion storage failed');
  }
  await FileSystem.deleteAsync(`${folder}${record.id}.json`,{idempotent:true});check();
  for(const item of record.result.detectedItems){
    await FileSystem.deleteAsync(`${folder}${record.id}.${item.reviewItemId}.saved`,{idempotent:true});check();
  }
}
export function commitScanReview(owner:string, result:AnalysisResult, sourceQueueId?:string):Promise<AnalysisResult> {
  if(Platform.OS==='web')return Promise.resolve(result); // Web review durability is a separate unfinished platform requirement.
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();const reviewId=id();
    const committed={...result,reviewId,sourceQueueId,detectedItems:result.detectedItems.map(item=>({...item,reviewItemId:id()}))};
    const record:RecordData={version:1,id:reviewId,owner,createdAt:Date.now(),sourceQueueId,result:committed};
    const text=JSON.stringify(record);if(text.length>32*1024*1024)throw Error('Scan review too large');
    const temporary=`${folder}${reviewId}.pending`,destination=`${folder}${reviewId}.json`;
    // Never overwrite an existing review, even if ID generation collides.
    if((await FileSystem.getInfoAsync(destination)).exists)throw Error('Scan review identity collision');check();
    let moved=false;
    try{
      await FileSystem.makeDirectoryAsync(folder,{intermediates:true});check();
      await FileSystem.writeAsStringAsync(temporary,text);check();
      await FileSystem.moveAsync({from:temporary,to:destination});moved=true;check();
      await readRecord(folder,reviewId,owner);check();notify();return committed;
    }catch(error){await FileSystem.deleteAsync(temporary,{idempotent:true}).catch(()=>{});if(moved)await FileSystem.deleteAsync(destination,{idempotent:true}).catch(()=>{});throw error;}
  });
}
export function listScanReviews(owner:string, onUnavailable?:(count:number)=>void):Promise<ScanReviewSummary[]> {
  if(Platform.OS==='web')return Promise.resolve([]);
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();const info=await FileSystem.getInfoAsync(folder);check();if(!info.exists)return[];
    const names=await FileSystem.readDirectoryAsync(folder);check();const reviews=[];let unavailable=0;
    for(const name of names){
      const reviewId=name.endsWith('.json')?name.slice(0,-5):'';if(!UUID.test(reviewId))continue;
      let record:RecordData;
      try { record=await readRecord(folder,reviewId,owner); }
      catch(error) { check(); if(error instanceof InvalidScanReview){unavailable++;continue;} throw error; }
      check();const pending=await pendingResult(folder,record);check();
      if(pending.detectedItems.length)reviews.push({id:reviewId,count:pending.detectedItems.length,createdAt:record.createdAt});
      else {
        // A previous acknowledgement may have succeeded before cleanup failed.
        // Retain the committed record on failure so a later refresh can retry.
        try { await finishReview(folder,record,check); }
        catch { check(); console.warn('Completed scan review cleanup pending'); }
      }
    }
    check();onUnavailable?.(unavailable);
    return reviews.sort((a,b)=>a.createdAt-b.createdAt);
  });
}
export function loadScanReview(owner:string,reviewId:string):Promise<AnalysisResult> {
  const {folder,check}=scope(owner);
  return serial(async()=>{check();const record=await readRecord(folder,reviewId,owner);check();const result=await pendingResult(folder,record);check();return result;});
}
async function markSaved(folder:string,reviewId:string,itemId:string,check:()=>void) {
  await FileSystem.writeAsStringAsync(`${folder}${reviewId}.${itemId}.saved`,'saved');check();
  if(await FileSystem.readAsStringAsync(`${folder}${reviewId}.${itemId}.saved`)!=='saved')throw Error('Scan acknowledgement storage failed');
  check();
}
/** Find pending review IDs confirmed by a complete authoritative wardrobe read. */
export function findConfirmedScanReviewItems(owner:string,itemIds:readonly string[]):Promise<string[]> {
  if(Platform.OS==='web')return Promise.resolve([]);
  const confirmed=new Set(itemIds.filter(value=>UUID.test(value)));
  if(!confirmed.size)return Promise.resolve([]);
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();if(!(await FileSystem.getInfoAsync(folder)).exists){check();return [];}check();
    const names=await FileSystem.readDirectoryAsync(folder);check();const matched=new Set<string>();
    for(const name of names){
      const reviewId=name.endsWith('.json')?name.slice(0,-5):'';if(!UUID.test(reviewId))continue;
      let record:RecordData;
      try{record=await readRecord(folder,reviewId,owner);}catch(error){check();if(error instanceof InvalidScanReview)continue;throw error;}
      check();const pending=await pendingResult(folder,record);check();
      for(const item of pending.detectedItems)if(confirmed.has(item.reviewItemId!))matched.add(item.reviewItemId!);
    }
    return [...matched];
  });
}
/** Reconcile cloud-confirmed garment IDs after a durable wardrobe acknowledgement. */
export function acknowledgeSavedScanReviewItems(owner:string,itemIds:readonly string[]):Promise<void> {
  if(Platform.OS==='web')return Promise.resolve();
  const saved=new Set(itemIds.filter(value=>UUID.test(value)));
  if(!saved.size)return Promise.resolve();
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();if(!(await FileSystem.getInfoAsync(folder)).exists){check();return;}check();
    const names=await FileSystem.readDirectoryAsync(folder);check();let changed=false;
    for(const name of names){
      const reviewId=name.endsWith('.json')?name.slice(0,-5):'';if(!UUID.test(reviewId))continue;
      let record:RecordData;
      try{record=await readRecord(folder,reviewId,owner);}catch(error){check();if(error instanceof InvalidScanReview)continue;throw error;}
      check();const matched=record.result.detectedItems.filter(item=>saved.has(item.reviewItemId!));
      if(!matched.length)continue;
      for(const item of matched)await markSaved(folder,reviewId,item.reviewItemId!,check);
      const pending=await pendingResult(folder,record);check();
      if(!pending.detectedItems.length)await finishReview(folder,record,check);
      changed=true;
    }
    if(changed)notify();
  });
}
export function acknowledgeScanReviewItem(owner:string,reviewId:string,itemId:string):Promise<void> {
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();if(!UUID.test(reviewId)||!UUID.test(itemId))throw Error('Invalid scan review item');
    // A sync retry may already have completed and removed the review.
    if(!(await FileSystem.getInfoAsync(`${folder}${reviewId}.json`)).exists){check();return;}check();
    const record=await readRecord(folder,reviewId,owner);check();
    if(!record.result.detectedItems.some(item=>item.reviewItemId===itemId))throw Error('Invalid scan review item');
    await markSaved(folder,reviewId,itemId,check);
    const pending=await pendingResult(folder,record);check();
    if(!pending.detectedItems.length)await finishReview(folder,record,check);
    notify();
  });
}
export function clearAllScanReviews(expectedRevision:number):Promise<void> {
  return serial(async()=>{if(expectedRevision!==authEvents.getRevision()||!FileSystem.documentDirectory)return;await FileSystem.deleteAsync(`${FileSystem.documentDirectory}${ROOT}/`,{idempotent:true});notify();});
}

export function findScanReviewBySource(owner:string,sourceQueueId:string):Promise<AnalysisResult|null>{
  if(Platform.OS==='web')return Promise.resolve(null);
  const {folder,check}=scope(owner);
  return serial(async()=>{
    check();if(!/^[a-z0-9_-]{1,100}$/i.test(sourceQueueId))throw Error('Invalid scan queue identity');
    const info=await FileSystem.getInfoAsync(folder);check();if(!info.exists)return null;
    const completed=await FileSystem.getInfoAsync(`${folder}${sourceQueueId}.done`);check();
    if(completed.exists){
      try{const value=JSON.parse(await FileSystem.readAsStringAsync(`${folder}${sourceQueueId}.done`));check();
        if(value.owner===owner&&value.sourceQueueId===sourceQueueId)return{sourceQueueId,detectedItems:[]};
      }catch{check();} // An interrupted marker cannot override an intact review record.
    }
    const names=await FileSystem.readDirectoryAsync(folder);check();
    for(const name of names){const reviewId=name.endsWith('.json')?name.slice(0,-5):'';if(!UUID.test(reviewId))continue;
      const record=await readRecord(folder,reviewId,owner);check();
      if(record.sourceQueueId===sourceQueueId){const result=await pendingResult(folder,record);check();return result;}
    }
    return null;
  });
}
