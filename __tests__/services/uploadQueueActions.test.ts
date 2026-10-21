jest.mock('../../src/services/scanReviews',()=>({commitScanReview:jest.fn(async (_owner:any,result:any)=>result),findScanReviewBySource:jest.fn().mockResolvedValue(null)}));
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { authEvents } from '../../store/authEvents';
import queue, { resetUploadQueueState, flushUploadQueuePersistence } from '../../src/store/uploadQueueStore';
import processor from '../../src/services/uploadQueueProcessor';
import { removePendingScanMedia } from '../../src/services/pendingScanMedia';
jest.mock('@react-native-community/netinfo', () => ({ __esModule:true, default:{fetch:jest.fn(),addEventListener:jest.fn(()=>jest.fn())} }));
jest.mock('../../src/services/pendingScanMedia', () => ({resolvePendingScanMedia:(uri:string)=>uri,preparePendingScanRemoval:jest.fn().mockResolvedValue(undefined),cancelPendingScanRemoval:jest.fn().mockResolvedValue(undefined),reconcilePendingScanMedia:jest.fn().mockResolvedValue(undefined),removePendingScanMedia:jest.fn().mockResolvedValue(undefined)}));
beforeEach(async () => {
  processor.destroy(); authEvents.emitLogin('owner'); resetUploadQueueState(); await flushUploadQueuePersistence();
  jest.clearAllMocks(); (NetInfo.fetch as jest.Mock).mockResolvedValue({isConnected:false});
  (removePendingScanMedia as jest.Mock).mockReset().mockResolvedValue(undefined);
});
afterEach(async () => { processor.destroy(); await flushUploadQueuePersistence().catch(()=>{}); });
async function failed() {
  queue.getState().addUpload('file:///managed-photo.jpg','image');
  queue.setState(state=>({pendingUploads:state.pendingUploads.map(item=>({...item,status:'failed',retryCount:3,error:'analysis_not_confirmed'}))}));
  await flushUploadQueuePersistence();return queue.getState().pendingUploads[0];
}
it('manual retry restores an exhausted owned entry to a durable pending state while offline', async () => {
  const entry=await failed();expect(await processor.retryUpload(entry.id)).toBe(true);
  expect(queue.getState().pendingUploads[0]).toMatchObject({status:'pending',retryCount:0});
  expect(JSON.parse((await AsyncStorage.getItem('wardrobe-upload-queue'))!).state.pendingUploads[0].retryCount).toBe(0);
});
it('manual retry can publish one successful result and remove its entry', async () => {
  const entry=await failed();(NetInfo.fetch as jest.Mock).mockResolvedValue({isConnected:true});
  const analyzeImage=jest.fn().mockResolvedValue(true);processor.setAnalysisHook({analyzeImage});
  expect(await processor.retryUpload(entry.id)).toBe(true);expect(analyzeImage).toHaveBeenCalledTimes(1);
  expect(queue.getState().pendingUploads).toEqual([]);
});
it.each(['retry','discard'])('another account cannot %s an owned queued scan', async action => {
  const entry=await failed();authEvents.emitLogin('other');
  expect(await (action==='retry'?processor.retryUpload(entry.id):processor.discardUpload(entry.id))).toBe(false);
  expect(queue.getState().pendingUploads[0]).toEqual(entry);expect(removePendingScanMedia).not.toHaveBeenCalled();
});
it('discard acknowledges queue removal before deleting the managed file', async () => {
  const entry=await failed();
  (removePendingScanMedia as jest.Mock).mockImplementationOnce(async()=>{
    expect(queue.getState().pendingUploads).toEqual([]);
    expect(JSON.parse((await AsyncStorage.getItem('wardrobe-upload-queue'))!).state.pendingUploads).toEqual([]);
  });
  expect(await processor.discardUpload(entry.id)).toBe(true);
  expect(removePendingScanMedia).toHaveBeenCalledWith('owner',entry.uri,expect.any(Function));
});
it.each(['retry','discard'])('failed %s persistence retains the prior recoverable entry', async action => {
  const entry=await failed();(AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
  await expect(action==='retry'?processor.retryUpload(entry.id):processor.discardUpload(entry.id)).rejects.toThrow('disk full');
  expect(queue.getState().pendingUploads[0]).toEqual(entry);
  expect(removePendingScanMedia).not.toHaveBeenCalled();
});
it('failed managed-file deletion restores the discarded record', async () => {
  const entry=await failed();(removePendingScanMedia as jest.Mock).mockRejectedValueOnce(new Error('delete failed'));
  await expect(processor.discardUpload(entry.id)).rejects.toThrow('delete failed');
  expect(queue.getState().pendingUploads[0]).toEqual(entry);
});
it('actions cannot reset or remove an entry while analysis is in flight', async () => {
  const entry=await failed();queue.getState().retryUpload(entry.id);
  (NetInfo.fetch as jest.Mock).mockResolvedValue({isConnected:true});
  let finish!:()=>void;processor.setAnalysisHook({analyzeImage:()=>new Promise(resolve=>{finish=()=>resolve(false);})});
  const work=processor.processQueue();for(let i=0;i<10;i++)await Promise.resolve();
  expect(await processor.retryUpload(entry.id)).toBe(false);expect(await processor.discardUpload(entry.id)).toBe(false);
  finish();await work;expect(queue.getState().pendingUploads).toHaveLength(1);
});
it('an account change during persistence cannot delete media or overwrite the next account queue', async () => {
  const entry=await failed();let finish!:(value:any)=>void;
  (AsyncStorage.setItem as jest.Mock).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
  const action=processor.discardUpload(entry.id);
  for(let i=0;i<10;i++)await Promise.resolve();
  expect(await processor.retryUpload(entry.id)).toBe(false);
  authEvents.emitLogin('other');queue.setState({pendingUploads:[]});queue.getState().addUpload('file:///other.jpg','image');
  const next=queue.getState().pendingUploads[0];finish(undefined);
  expect(await action).toBe(false);await flushUploadQueuePersistence();
  expect(queue.getState().pendingUploads).toEqual([next]);expect(removePendingScanMedia).not.toHaveBeenCalled();
});
