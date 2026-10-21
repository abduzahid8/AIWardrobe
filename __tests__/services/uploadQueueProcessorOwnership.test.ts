jest.mock('../../src/services/scanReviews',()=>({commitScanReview:jest.fn(async (_owner:any,result:any)=>result),findScanReviewBySource:jest.fn().mockResolvedValue(null)}));
import { findScanReviewBySource } from '../../src/services/scanReviews';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { authEvents } from '../../store/authEvents';
import useUploadQueueStore, { resetUploadQueueState, flushUploadQueuePersistence } from '../../src/store/uploadQueueStore';
import processor from '../../src/services/uploadQueueProcessor';
jest.mock('@react-native-community/netinfo', () => ({ __esModule: true, default: { fetch: jest.fn(), addEventListener: jest.fn(() => jest.fn()) } }));
function deferred() { let resolve!: (value: any) => void; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; }
beforeEach(async () => {
  processor.destroy(); authEvents.emitLogin('owner'); resetUploadQueueState(); await flushUploadQueuePersistence();
  jest.clearAllMocks(); (NetInfo.fetch as jest.Mock).mockResolvedValue({ isConnected: true, isInternetReachable: true });
});
afterEach(async () => { processor.destroy(); await flushUploadQueuePersistence(); });
const enqueue = () => { useUploadQueueStore.getState().addUpload('file:///private.jpg', 'image'); return useUploadQueueStore.getState().pendingUploads[0]; };
it('explicit successful analysis removes only its current-owner entry', async () => {
  enqueue(); const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage });
  await processor.processQueue(); expect(analyzeImage).toHaveBeenCalledTimes(1); expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});
it.each([undefined, false])('unconfirmed result %j never silently deletes a failed scan', async value => {
  enqueue(); processor.setAnalysisHook({ analyzeImage: jest.fn().mockResolvedValue(value) }); await processor.processQueue();
  expect(useUploadQueueStore.getState().pendingUploads[0]).toMatchObject({ status: 'pending', retryCount: 1 });
});
it('exhausted entry stops after three attempts rather than looping forever', async () => {
  enqueue(); const analyzeImage = jest.fn().mockResolvedValue(false); processor.setAnalysisHook({ analyzeImage });
  const clock=jest.spyOn(Date,'now');
  try { for (let i = 0; i < 5; i++) { clock.mockReturnValue(1_000_000+i*300_000); await processor.processQueue(); } }
  finally { clock.mockRestore(); }
  expect(analyzeImage).toHaveBeenCalledTimes(3); expect(useUploadQueueStore.getState().pendingUploads[0]).toMatchObject({ status: 'failed', retryCount: 3 });
});
it('another owner cannot process an earlier account entry', async () => {
  const item = enqueue(); authEvents.emitLogin('other'); const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage });
  await processor.processQueue(); expect(analyzeImage).not.toHaveBeenCalled(); expect(useUploadQueueStore.getState().pendingUploads[0]).toEqual(item);
});
it('an old-account hook cannot process a new owner entry', async () => {
  const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage }); authEvents.emitLogin('other'); enqueue();
  await processor.processQueue(); expect(analyzeImage).not.toHaveBeenCalled(); expect(useUploadQueueStore.getState().pendingUploads[0].status).toBe('pending');
});
it('two network checks cannot start duplicate processing loops', async () => {
  enqueue(); const network = deferred(); (NetInfo.fetch as jest.Mock).mockReturnValueOnce(network.promise);
  const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage });
  const first = processor.processQueue(), second = processor.processQueue(); network.resolve({ isConnected: true }); await Promise.all([first, second]);
  expect(NetInfo.fetch).toHaveBeenCalledTimes(1); expect(analyzeImage).toHaveBeenCalledTimes(1);
});
it.each(['account', 'destroy'])('%s invalidates an in-flight success without deleting private entries', async action => {
  enqueue(); const work = deferred(); processor.setAnalysisHook({ analyzeImage: () => work.promise });
  const request = processor.processQueue(); await Promise.resolve();
  if (action === 'account') authEvents.emitLogin('other'); else processor.destroy();
  work.resolve(true); await request; expect(useUploadQueueStore.getState().pendingUploads).toHaveLength(1);
});

it('connectivity lookup failure retains entries and releases the processing lock', async () => {
  enqueue(); (NetInfo.fetch as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage });
  await processor.processQueue(); expect(useUploadQueueStore.getState().pendingUploads).toHaveLength(1);
  await processor.processQueue(); expect(analyzeImage).toHaveBeenCalledTimes(1); expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});

it('busy review screen prevents analysis or attempt consumption', async () => {
  const entry = enqueue(); const analyzeImage = jest.fn().mockResolvedValue(true);
  processor.setAnalysisHook({ analyzeImage, canProcess: () => false }); await processor.processQueue();
  expect(analyzeImage).not.toHaveBeenCalled(); expect(useUploadQueueStore.getState().pendingUploads[0]).toEqual(entry);
});
it('explicitly unreachable internet does not consume attempts', async () => {
  const entry = enqueue(); (NetInfo.fetch as jest.Mock).mockResolvedValueOnce({ isConnected: true, isInternetReachable: false });
  const analyzeImage = jest.fn().mockResolvedValue(true); processor.setAnalysisHook({ analyzeImage }); await processor.processQueue();
  expect(analyzeImage).not.toHaveBeenCalled(); expect(useUploadQueueStore.getState().pendingUploads[0]).toEqual(entry);
});
it('initialization recovers interrupted current-owner work for a later attempt', async () => {
  const entry = enqueue(); useUploadQueueStore.getState().updateStatus(entry.id, 'processing');
  processor.setAnalysisHook({ canProcess: () => false }); processor.initialize(); await Promise.resolve();
  expect(useUploadQueueStore.getState().pendingUploads[0].status).toBe('pending');
});

it('late persisted hydration recovers interrupted scans and starts only after it finishes', async () => {
  let finishHydration!:()=>void;
  const hydrated=jest.spyOn(useUploadQueueStore.persist,'hasHydrated').mockReturnValue(false);
  const unsubscribe=jest.fn();
  const onFinish=jest.spyOn(useUploadQueueStore.persist,'onFinishHydration').mockImplementation(callback=>{finishHydration=()=>callback(useUploadQueueStore.getState());return unsubscribe;});
  try {
    const item=enqueue();useUploadQueueStore.getState().updateStatus(item.id,'processing');
    const analyzeImage=jest.fn().mockResolvedValue(true);processor.setAnalysisHook({analyzeImage});processor.initialize();
    expect(NetInfo.fetch).not.toHaveBeenCalled();expect(analyzeImage).not.toHaveBeenCalled();
    hydrated.mockReturnValue(true);finishHydration();
    for(let i=0;i<30;i++)await Promise.resolve();
    expect(analyzeImage).toHaveBeenCalledTimes(1);
    expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
    processor.destroy();expect(unsubscribe).toHaveBeenCalledTimes(1);
  } finally {hydrated.mockRestore();onFinish.mockRestore();}
});

it('durably completed source queues are cleaned offline even after retries are exhausted', async () => {
  const entry=enqueue();useUploadQueueStore.setState(state=>({pendingUploads:state.pendingUploads.map(item=>({...item,status:'failed',retryCount:3,nextRetryAt:Date.now()+300_000}))}));
  (NetInfo.fetch as jest.Mock).mockResolvedValue({isConnected:false});
  (findScanReviewBySource as jest.Mock).mockResolvedValueOnce({sourceQueueId:entry.id,detectedItems:[]});
  const analyzeImage=jest.fn(),restoreReview=jest.fn();processor.setAnalysisHook({analyzeImage,restoreReview,canProcess:()=>false});
  await processor.processQueue();expect(analyzeImage).not.toHaveBeenCalled();expect(restoreReview).not.toHaveBeenCalled();expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});

it('persists retry cooldown and repeated network checks cannot consume attempts before it expires', async()=>{
 enqueue();let now=1_000_000;const clock=jest.spyOn(Date,'now').mockImplementation(()=>now);
 const analyzeImage=jest.fn().mockResolvedValue(false);processor.setAnalysisHook({analyzeImage});
 try {
  await processor.processQueue();
  expect(useUploadQueueStore.getState().pendingUploads[0]).toMatchObject({retryCount:1,nextRetryAt:1_030_000});
  const raw=await AsyncStorage.getItem('wardrobe-upload-queue');expect(raw).toContain('1030000');
  processor.destroy();processor.setAnalysisHook({analyzeImage});
  for(let i=0;i<10;i++)await processor.processQueue();expect(analyzeImage).toHaveBeenCalledTimes(1);
  now=1_030_000;await processor.processQueue();expect(analyzeImage).toHaveBeenCalledTimes(2);
  expect(useUploadQueueStore.getState().pendingUploads[0]).toMatchObject({retryCount:2,nextRetryAt:1_150_000});
 } finally { clock.mockRestore(); }
});
it('an outage stops after one failed capture rather than draining attempts across the whole queue',async()=>{
 enqueue();useUploadQueueStore.getState().addUpload('file:///another.jpg','image');
 const analyzeImage=jest.fn().mockResolvedValue(false);processor.setAnalysisHook({analyzeImage});
 await processor.processQueue();expect(analyzeImage).toHaveBeenCalledTimes(1);
 for(let i=0;i<10;i++)await processor.processQueue();
 expect(analyzeImage).toHaveBeenCalledTimes(1);
 expect(useUploadQueueStore.getState().pendingUploads.map(item=>item.retryCount)).toEqual([1,0]);
});
it('explicit retry clears the cooldown and can analyze immediately',async()=>{
 const entry=enqueue();const analyzeImage=jest.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true);processor.setAnalysisHook({analyzeImage});
 await processor.processQueue();expect(useUploadQueueStore.getState().pendingUploads[0].nextRetryAt).toBeGreaterThan(Date.now());
 expect(await processor.retryUpload(entry.id)).toBe(true);expect(analyzeImage).toHaveBeenCalledTimes(2);expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});

it('a previous owner cooldown cannot delay the current owner capture',async()=>{
 enqueue();const analyzeImage=jest.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true);processor.setAnalysisHook({analyzeImage});
 await processor.processQueue();const former=useUploadQueueStore.getState().pendingUploads[0];
 authEvents.emitLogin('other');enqueue();processor.setAnalysisHook({analyzeImage});
 await processor.processQueue();expect(analyzeImage).toHaveBeenCalledTimes(2);
 expect(useUploadQueueStore.getState().pendingUploads).toEqual([former]);
});
