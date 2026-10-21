import { Platform } from 'react-native';
import * as FS from 'expo-file-system/legacy';
import { authEvents } from '../../store/authEvents';
import { findConfirmedScanReviewItems,acknowledgeSavedScanReviewItems,commitScanReview,listScanReviews,loadScanReview,acknowledgeScanReviewItem,findScanReviewBySource,clearAllScanReviews,subscribeScanReviews } from '../../src/services/scanReviews';
jest.mock('expo-file-system/legacy',()=>({documentDirectory:'file:///documents/',getInfoAsync:jest.fn(),makeDirectoryAsync:jest.fn(),readAsStringAsync:jest.fn(),writeAsStringAsync:jest.fn(),moveAsync:jest.fn(),deleteAsync:jest.fn(),readDirectoryAsync:jest.fn()}));
const platformDescriptor=Object.getOwnPropertyDescriptor(Platform,'OS')!;
afterEach(()=>Object.defineProperty(Platform,'OS',platformDescriptor));
const owner='11111111-1111-4111-8111-111111111111',other='22222222-2222-4222-8222-222222222222';
let files:Map<string,string>,dirs:Set<string>;
const result:any={detectedItems:[0,1].map(index=>({itemType:index?'pants':'jacket',color:'beige',style:'casual',description:'Original result',position:index?'lower':'upper',confidence:0,confidenceLevel:'low',detectionSources:['nvidia_classify'],frameImage:'data:image/jpeg;base64,AA=='}))};
beforeEach(()=>{
 Object.defineProperty(Platform,'OS',{configurable:true,value:'ios'});authEvents.emitLogin(owner);files=new Map();dirs=new Set();jest.clearAllMocks();
 (FS.getInfoAsync as jest.Mock).mockImplementation(async uri=>({exists:files.has(uri)||dirs.has(uri),isDirectory:dirs.has(uri),size:files.has(uri)?Buffer.byteLength(files.get(uri)!):0}));
 (FS.makeDirectoryAsync as jest.Mock).mockImplementation(async uri=>{dirs.add(uri);});
 (FS.writeAsStringAsync as jest.Mock).mockImplementation(async(uri,text)=>{files.set(uri,text);});
 (FS.readAsStringAsync as jest.Mock).mockImplementation(async uri=>{if(!files.has(uri))throw Error('missing');return files.get(uri);});
 (FS.moveAsync as jest.Mock).mockImplementation(async({from,to})=>{files.set(to,files.get(from)!);files.delete(from);});
 (FS.deleteAsync as jest.Mock).mockImplementation(async uri=>{for(const path of files.keys())if(path===uri||path.startsWith(uri))files.delete(path);});
 (FS.readDirectoryAsync as jest.Mock).mockImplementation(async uri=>[...files.keys()].filter(path=>path.startsWith(uri)).map(path=>path.slice(uri.length)).filter(path=>!path.includes('/')));
});
it('commits complete image bytes and metadata before exposing a restorable review',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 expect(saved.reviewId).toBeDefined();expect(saved.detectedItems[0].reviewItemId).toBeDefined();
 const loaded=await loadScanReview(owner,saved.reviewId!);expect(loaded).toEqual(saved);
 expect(await listScanReviews(owner)).toEqual([expect.objectContaining({id:saved.reviewId,count:2})]);
 expect(await findScanReviewBySource(owner,'queue1')).toEqual(saved);
});
it('acknowledges individual items and retains a completion tombstone for an interrupted source queue',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 await acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!);
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toEqual([saved.detectedItems[1]]);
 await acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[1].reviewItemId!);
 expect(await listScanReviews(owner)).toEqual([]);
 expect(await findScanReviewBySource(owner,'queue1')).toEqual({sourceQueueId:'queue1',detectedItems:[]});
 expect([...files.keys()].filter(path=>path.endsWith('.json'))).toEqual([]);
});
it('another account cannot load the former owner review',async()=>{
 const saved=await commitScanReview(owner,result);authEvents.emitLogin(other);
 expect(()=>loadScanReview(owner,saved.reviewId!)).toThrow('account changed');expect(await listScanReviews(other)).toEqual([]);
});
it('failed persistence cannot expose a successful review',async()=>{
 (FS.writeAsStringAsync as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(commitScanReview(owner,result)).rejects.toThrow('disk full');expect(await listScanReviews(owner)).toEqual([]);
});
it('an account change during file write aborts and removes only the uncommitted result',async()=>{
 (FS.writeAsStringAsync as jest.Mock).mockImplementationOnce(async(uri,text)=>{files.set(uri,text);authEvents.emitLogin(other);});
 await expect(commitScanReview(owner,result)).rejects.toThrow('account changed');expect([...files.keys()]).toEqual([]);
});
it('a failing observer cannot erase a committed review',async()=>{
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});const stop=subscribeScanReviews(()=>{throw Error('observer');});
 try{const saved=await commitScanReview(owner,result);expect(await loadScanReview(owner,saved.reviewId!)).toEqual(saved);}finally{stop();warn.mockRestore();}
});
it('account cleanup removes review images, acknowledgements and source tombstones',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');await acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!);
 await clearAllScanReviews(authEvents.getRevision());expect(files.size).toBe(0);
});
it('an interrupted empty acknowledgement cannot discard an unsaved candidate',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 const marker=`file:///documents/aiwardrobe-scan-reviews/${owner}/${saved.reviewId}.${saved.detectedItems[0].reviewItemId}.saved`;
 files.set(marker,'');
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toHaveLength(2);
 expect((await listScanReviews(owner))[0].count).toBe(2);
});
it('retries completed review cleanup after a deletion failure without losing the source completion marker',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 await acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!);
 (FS.deleteAsync as jest.Mock).mockRejectedValueOnce(Error('busy file'));
 await expect(acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[1].reviewItemId!)).rejects.toThrow('busy file');
 expect([...files.keys()].some(path=>path.endsWith('.json'))).toBe(true);
 expect(await listScanReviews(owner)).toEqual([]);
 expect([...files.keys()].some(path=>path.endsWith('.json'))).toBe(false);
 expect(await findScanReviewBySource(owner,'queue1')).toEqual({sourceQueueId:'queue1',detectedItems:[]});
});
it('does not delete the review when the completion marker cannot be read back',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 await acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!);
 const original=(FS.writeAsStringAsync as jest.Mock).getMockImplementation()!;
 (FS.writeAsStringAsync as jest.Mock).mockImplementation(async(uri,text)=>original(uri,uri.endsWith('.done')?'{}':text));
 await expect(acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[1].reviewItemId!)).rejects.toThrow('storage failed');
 expect([...files.keys()].some(path=>path.endsWith('.json'))).toBe(true);
 (FS.writeAsStringAsync as jest.Mock).mockImplementation(original);
 expect(await listScanReviews(owner)).toEqual([]);
 expect(await findScanReviewBySource(owner,'queue1')).toEqual({sourceQueueId:'queue1',detectedItems:[]});
});
it('a persistent cleanup failure does not hide another pending review',async()=>{
 const completed=await commitScanReview(owner,result,'queue1');
 const pending=await commitScanReview(owner,result,'queue2');
 await acknowledgeScanReviewItem(owner,completed.reviewId!,completed.detectedItems[0].reviewItemId!);
 (FS.deleteAsync as jest.Mock).mockRejectedValue(Error('busy file'));
 await expect(acknowledgeScanReviewItem(owner,completed.reviewId!,completed.detectedItems[1].reviewItemId!)).rejects.toThrow('busy file');
 const warn=jest.spyOn(console,'warn').mockImplementation(()=>{});
 try { expect(await listScanReviews(owner)).toEqual([expect.objectContaining({id:pending.reviewId,count:2})]); }
 finally { warn.mockRestore(); }
 expect((await loadScanReview(owner,pending.reviewId!)).detectedItems).toHaveLength(2);
});
it.each(['{','null',JSON.stringify({version:1,result:{detectedItems:[null]}})])('isolates a corrupt record without deleting it or hiding valid reviews (%s)',async bad=>{
 const saved=await commitScanReview(owner,result,'queue1');
 const damaged=`file:///documents/aiwardrobe-scan-reviews/${owner}/33333333-3333-4333-8333-333333333333.json`;
 if(bad.startsWith('{"version"'))bad=JSON.stringify({...JSON.parse(files.get(`file:///documents/aiwardrobe-scan-reviews/${owner}/${saved.reviewId}.json`)! ),id:'33333333-3333-4333-8333-333333333333',result:{detectedItems:[null]}});
 files.set(damaged,bad);const unavailable=jest.fn();
 expect(await listScanReviews(owner,unavailable)).toEqual([expect.objectContaining({id:saved.reviewId,count:2})]);
 expect(unavailable).toHaveBeenCalledWith(1);expect(files.get(damaged)).toBe(bad);
});
it('a transient filesystem error is reported rather than classified as record corruption',async()=>{
 await commitScanReview(owner,result,'queue1');
 (FS.readAsStringAsync as jest.Mock).mockRejectedValueOnce(Error('device unavailable'));const unavailable=jest.fn();
 await expect(listScanReviews(owner,unavailable)).rejects.toThrow('device unavailable');expect(unavailable).not.toHaveBeenCalled();
});
it.each(['duplicate item IDs','invalid timestamp','invalid source identity'])('rejects %s without acknowledging or deleting the original record',async fault=>{
 const saved=await commitScanReview(owner,result,'queue1');
 const uri=`file:///documents/aiwardrobe-scan-reviews/${owner}/${saved.reviewId}.json`;
 const record=JSON.parse(files.get(uri)!);
 if(fault==='duplicate item IDs')record.result.detectedItems[1].reviewItemId=record.result.detectedItems[0].reviewItemId;
 if(fault==='invalid timestamp')record.createdAt='yesterday';
 if(fault==='invalid source identity')record.sourceQueueId='../queue1';
 const damaged=JSON.stringify(record);files.set(uri,damaged);
 await expect(acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!)).rejects.toThrow();
 expect(files.get(uri)).toBe(damaged);expect([...files.keys()].some(path=>path.endsWith('.saved'))).toBe(false);
 const unavailable=jest.fn();expect(await listScanReviews(owner,unavailable)).toEqual([]);expect(unavailable).toHaveBeenCalledWith(1);
});
it('does not overwrite an existing review when review ID generation collides',async()=>{
 const first=await commitScanReview(owner,{...result,detectedItems:[result.detectedItems[0]]});
 const uri=`file:///documents/aiwardrobe-scan-reviews/${owner}/${first.reviewId}.json`;
 const original=files.get(uri)!;
 // Reproduce the same UUID nibble sequence for the next review ID.
 const digits=first.reviewId!.replace(/-/g,'').split('');let cursor=0;
 const random=jest.spyOn(Math,'random').mockImplementation(()=>{
  // The generator does not call random for the fixed version nibble.
  if(cursor===12)cursor++;
  return parseInt(digits[cursor++]||'0',16)/16;
 });
 try { await expect(commitScanReview(owner,result)).rejects.toThrow('identity collision'); }
 finally { random.mockRestore(); }
 expect(files.get(uri)).toBe(original);expect(await loadScanReview(owner,first.reviewId!)).toEqual(first);
});
it('cannot publish a review when generated candidate IDs collide',async()=>{
 const random=jest.spyOn(Math,'random').mockReturnValue(0);
 try { await expect(commitScanReview(owner,result)).rejects.toThrow('Duplicate scan review item'); }
 finally { random.mockRestore(); }
 expect(files.size).toBe(0);expect(await listScanReviews(owner)).toEqual([]);
});
it('replaying final acknowledgement after a completed review is harmless',async()=>{
 const saved=await commitScanReview(owner,{...result,detectedItems:[result.detectedItems[0]]},'queue1');
 const item=saved.detectedItems[0].reviewItemId!;
 await acknowledgeScanReviewItem(owner,saved.reviewId!,item);
 await expect(acknowledgeScanReviewItem(owner,saved.reviewId!,item)).resolves.toBeUndefined();
 expect(await findScanReviewBySource(owner,'queue1')).toEqual({sourceQueueId:'queue1',detectedItems:[]});
});
it('an acknowledgement cannot succeed when its saved marker was not written correctly',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 (FS.writeAsStringAsync as jest.Mock).mockResolvedValueOnce(undefined);
 await expect(acknowledgeScanReviewItem(owner,saved.reviewId!,saved.detectedItems[0].reviewItemId!)).rejects.toThrow();
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toHaveLength(2);
});
it('cloud-confirmed stable garment IDs remove only their matching scan-review candidates',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 await acknowledgeSavedScanReviewItems(owner,[saved.detectedItems[0].reviewItemId!,'99999999-9999-4999-8999-999999999999']);
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toEqual([saved.detectedItems[1]]);
 await acknowledgeSavedScanReviewItems(owner,[saved.detectedItems[1].reviewItemId!]);
 expect(await listScanReviews(owner)).toEqual([]);
 expect(await findScanReviewBySource(owner,'queue1')).toEqual({sourceQueueId:'queue1',detectedItems:[]});
 await expect(acknowledgeSavedScanReviewItems(owner,saved.detectedItems.map(i=>i.reviewItemId!))).resolves.toBeUndefined();
});
it('failed reconciliation keeps the source review available for journal replay',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 (FS.writeAsStringAsync as jest.Mock).mockRejectedValueOnce(Error('disk full'));
 await expect(acknowledgeSavedScanReviewItems(owner,[saved.detectedItems[0].reviewItemId!])).rejects.toThrow('disk full');
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toHaveLength(2);
 await acknowledgeSavedScanReviewItems(owner,[saved.detectedItems[0].reviewItemId!]);
 expect((await loadScanReview(owner,saved.reviewId!)).detectedItems).toEqual([saved.detectedItems[1]]);
});
it('legacy reconciliation finds only pending matching IDs and cannot remove photos during discovery',async()=>{
 const saved=await commitScanReview(owner,result,'queue1');
 const before=new Map(files);
 expect(await findConfirmedScanReviewItems(owner,[saved.detectedItems[0].reviewItemId!])).toEqual([saved.detectedItems[0].reviewItemId]);
 expect(files).toEqual(before);
 await acknowledgeSavedScanReviewItems(owner,[saved.detectedItems[0].reviewItemId!]);
 expect(await findConfirmedScanReviewItems(owner,saved.detectedItems.map(i=>i.reviewItemId!))).toEqual([saved.detectedItems[1].reviewItemId]);
 expect(await findConfirmedScanReviewItems(owner,['99999999-9999-4999-8999-999999999999'])).toEqual([]);
});
