jest.mock('../../src/services/scanReviews',()=>({commitScanReview:jest.fn(async (_owner:any,result:any)=>result),findScanReviewBySource:jest.fn().mockResolvedValue(null)}));
jest.mock('../../src/services/videoFrameSampling', () => ({ readVideoSampleTimes: jest.fn().mockResolvedValue([0, 1000, 2000]) }));
import { act, renderHook } from '@testing-library/react-native';
import { Alert, Platform } from 'react-native';
import { authEvents } from '../../store/authEvents';
import { useVideoAnalysis } from '../../src/features/wardrobe/useVideoAnalysis';
import { ExternalAIService } from '../../src/services/externalAIService';
import * as FileSystem from 'expo-file-system/legacy';
import * as VideoThumbnails from 'expo-video-thumbnails';
const mockAddUpload = jest.fn();
jest.mock('../../src/store/uploadQueueStore', () => ({ __esModule: true, default: { getState: () => ({ addUpload: mockAddUpload }) } }));
jest.mock('../../src/services/externalAIService', () => ({ ExternalAIService: { classifyOnly: jest.fn(), processVideoFrames: jest.fn() } }));
jest.mock('expo-video-thumbnails', () => ({ getThumbnailAsync: jest.fn() }));
jest.mock('expo-file-system/legacy', () => ({ readAsStringAsync: jest.fn(), deleteAsync: jest.fn().mockResolvedValue(undefined) }));
const descriptor = Object.getOwnPropertyDescriptor(Platform, 'OS')!;
function deferred() { let resolve!: (value: any) => void; let reject!: (error: Error) => void; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }
const detection = (category = 'jacket') => ({ success: true, imageUrl: `data:image/jpeg;base64,${category}`, steps: [], classification: { category, section: 'outerwear', confidence: 0.9, attributes: { color: 'beige', style: 'casual', material: null } } });
beforeEach(() => { jest.clearAllMocks(); authEvents.emitLogin('owner'); Object.defineProperty(Platform, 'OS', { configurable: true, value: 'web' }); });
afterEach(() => { Object.defineProperty(Platform, 'OS', descriptor); jest.restoreAllMocks(); });
it('older completion cannot replace the newer image result', async () => {
  const old = deferred(), next = deferred();
  (ExternalAIService.classifyOnly as jest.Mock).mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise);
  const { result } = renderHook(() => useVideoAnalysis()); let first!: Promise<boolean>, second!: Promise<boolean>;
  act(() => { first = result.current.analyzeImage('old'); second = result.current.analyzeImage('new'); });
  await act(async () => { next.resolve(detection('coat')); await second; });
  await act(async () => { old.resolve(detection()); await first; });
  expect(result.current.results?.detectedItems[0].itemType).toBe('coat');
});
it('older completion cannot clear the active request spinner', async () => {
  const old = deferred(), next = deferred();
  (ExternalAIService.classifyOnly as jest.Mock).mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise);
  const { result } = renderHook(() => useVideoAnalysis()); let first!: Promise<boolean>, second!: Promise<boolean>;
  act(() => { first = result.current.analyzeImage('old'); second = result.current.analyzeImage('new'); });
  await act(async () => { old.resolve(detection()); await first; });
  expect(result.current.analyzing).toBe(true); expect(result.current.results).toBeNull();
  await act(async () => { next.resolve(detection('coat')); await second; });
  expect(result.current.analyzing).toBe(false);
});
it.each(['reset', 'account', 'unmount'])('%s discards pending errors and never queues a stale photo', async mode => {
  const pending = deferred(); (ExternalAIService.classifyOnly as jest.Mock).mockReturnValueOnce(pending.promise);
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result, unmount } = renderHook(() => useVideoAnalysis()); let request!: Promise<boolean>;
  act(() => { request = result.current.analyzeImage('old-private-photo'); });
  act(() => { if (mode === 'reset') result.current.reset(); else if (mode === 'account') { authEvents.emitLogout(); authEvents.emitLogin('other'); } else unmount(); });
  await act(async () => { pending.reject(new Error('network timeout')); await request; });
  expect(alert).not.toHaveBeenCalled(); expect(mockAddUpload).not.toHaveBeenCalled();
  if (mode !== 'unmount') { expect(result.current.results).toBeNull(); expect(result.current.analyzing).toBe(false); }
});
it('account change immediately clears a completed private detection', async () => {
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValueOnce(detection());
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage('private-photo'); });
  expect(result.current.results).not.toBeNull();
  act(() => { authEvents.emitLogin('other'); });
  expect(result.current.results).toBeNull();
});
it('reset during native image reading prevents a subsequent provider call', async () => {
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'ios' });
  const read = deferred(); (FileSystem.readAsStringAsync as jest.Mock).mockReturnValueOnce(read.promise);
  const { result } = renderHook(() => useVideoAnalysis()); let request!: Promise<boolean>;
  act(() => { request = result.current.analyzeImage('file:///old.jpg'); result.current.reset(); });
  await act(async () => { read.resolve('photo'); await request; });
  expect(ExternalAIService.classifyOnly).not.toHaveBeenCalled();
});
it('reset during native video extraction prevents reading or sending old frames', async () => {
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'ios' });
  const frame = deferred(); (VideoThumbnails.getThumbnailAsync as jest.Mock).mockReturnValueOnce(frame.promise);
  const { result } = renderHook(() => useVideoAnalysis()); let request!: Promise<boolean>;
  act(() => { request = result.current.analyzeVideo('file:///old.mp4'); result.current.reset(); });
  await act(async () => { frame.resolve({ uri: 'file:///frame.jpg' }); await request; });
  expect(FileSystem.readAsStringAsync).not.toHaveBeenCalled(); expect(ExternalAIService.classifyOnly).not.toHaveBeenCalled();
});

it('a video batch completing after an account switch cannot publish old frame results', async () => {
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'ios' });
  (VideoThumbnails.getThumbnailAsync as jest.Mock).mockResolvedValue({ uri:'file:///frame.jpg' });
  (FileSystem.readAsStringAsync as jest.Mock).mockResolvedValue('frame');
  const batch = deferred();
  (ExternalAIService.processVideoFrames as jest.Mock).mockReturnValueOnce(batch.promise);
  const { result } = renderHook(() => useVideoAnalysis());
  let request!: Promise<boolean>;
  act(() => { request = result.current.analyzeVideo('file:///old-owner.mp4'); });
  await act(async () => { for(let i=0;i<20;i++) await Promise.resolve(); });
  expect(ExternalAIService.processVideoFrames).toHaveBeenCalledTimes(1);
  act(() => { authEvents.emitLogout(); authEvents.emitLogin('next-owner'); });
  await act(async () => { batch.resolve([detection()]); expect(await request).toBe(false); });
  expect(result.current.results).toBeNull();
  expect(mockAddUpload).not.toHaveBeenCalled();
});
it('a thumbnail returned after cancellation is removed without reading or publishing it', async () => {
  Object.defineProperty(Platform, 'OS', { configurable:true, value:'ios' });
  const thumbnail = deferred();
  (VideoThumbnails.getThumbnailAsync as jest.Mock).mockReset().mockReturnValueOnce(thumbnail.promise);
  const { result } = renderHook(() => useVideoAnalysis());
  let request!: Promise<boolean>;
  act(() => { request = result.current.analyzeVideo('file:///cancelled.mp4'); });
  await act(async () => { for(let i=0;i<10;i++) await Promise.resolve(); });
  expect(VideoThumbnails.getThumbnailAsync).toHaveBeenCalledTimes(1);
  act(() => { result.current.reset(); });
  await act(async () => { thumbnail.resolve({ uri:'file:///temporary-thumbnail.jpg' }); expect(await request).toBe(false); });
  expect(FileSystem.deleteAsync).toHaveBeenCalledWith('file:///temporary-thumbnail.jpg', { idempotent:true });
  expect(FileSystem.readAsStringAsync).not.toHaveBeenCalled();
  expect(result.current.results).toBeNull();
});
