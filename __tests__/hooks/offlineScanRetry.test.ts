jest.mock('../../src/services/scanReviews',()=>({commitScanReview:jest.fn(async (_owner:any,result:any)=>result),findScanReviewBySource:jest.fn().mockResolvedValue(null)}));
jest.mock('../../src/services/videoFrameSampling', () => ({ readVideoSampleTimes: jest.fn().mockResolvedValue([0, 1000, 2000]) }));
import { act, renderHook } from '@testing-library/react-native';
import * as pendingMedia from '../../src/services/pendingScanMedia';
import * as FileSystem from 'expo-file-system/legacy';
import { Alert, Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authEvents } from '../../store/authEvents';
import { useVideoAnalysis } from '../../src/features/wardrobe/useVideoAnalysis';
import { ExternalAIService } from '../../src/services/externalAIService';
import queue, { resetUploadQueueState, flushUploadQueuePersistence } from '../../src/store/uploadQueueStore';
import processor from '../../src/services/uploadQueueProcessor';
const ts = require('typescript');
const screenText = require('fs').readFileSync(require('path').resolve(__dirname, '../../screens/WardrobeVideoScreen.tsx'), 'utf8');
const screenAst = ts.createSourceFile('WardrobeVideoScreen.tsx', screenText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let screenBindingCode = '';
function visit(node: any) {
  if (ts.isCallExpression(node) && node.expression.getText(screenAst) === 'useEffect' && node.arguments[0]?.getText(screenAst).includes('uploadQueueProcessor.setAnalysisHook')) screenBindingCode = ts.transpile(`const binding = ${node.arguments[0].getText(screenAst)};`, { target: ts.ScriptTarget.ES2020 });
  ts.forEachChild(node, visit);
}
visit(screenAst);
jest.mock('../../src/services/externalAIService', () => ({ ExternalAIService: { classifyOnly: jest.fn() } }));
jest.mock('@react-native-community/netinfo', () => ({ __esModule: true, default: { fetch: jest.fn(async () => ({ isConnected: true, isInternetReachable: true })), addEventListener: jest.fn(() => jest.fn()) } }));
jest.mock('expo-video-thumbnails', () => ({ getThumbnailAsync: jest.fn() }));
jest.mock('expo-file-system/legacy', () => ({ readAsStringAsync: jest.fn() }));
const descriptor = Object.getOwnPropertyDescriptor(Platform, 'OS')!;
const photo = 'data:image/jpeg;base64,UEhPVE8=';
beforeEach(async () => { processor.destroy(); authEvents.emitLogin('owner'); resetUploadQueueState(); await flushUploadQueuePersistence(); jest.clearAllMocks(); Object.defineProperty(Platform, 'OS', { configurable: true, value: 'web' }); });
afterEach(async () => { processor.destroy(); await flushUploadQueuePersistence().catch(() => undefined); Object.defineProperty(Platform, 'OS', descriptor); jest.restoreAllMocks(); });
it('real worker and hook publish one queued result for review and preserve the next entry', async () => {
  queue.getState().addUpload(photo, 'image'); queue.getState().addUpload(`${photo}Mg==`, 'image');
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({ success: true, imageUrl: photo, steps: [], classification: { category: 'jacket', section: 'outerwear', confidence: 0.9, attributes: { color: 'beige', material: 'cotton', style: 'casual' } } });
  const { result } = renderHook(() => useVideoAnalysis());
  const bind = Function('uploadQueueProcessor', 'analyzeImage', 'analyzeVideo', 'initialMediaPending', 'queueGate', 'reviewRestorePending', 'navigation', 'restoreReview', `${screenBindingCode} return binding;`)(processor, result.current.analyzeImage, result.current.analyzeVideo, { current: false }, { get current() { return { analyzing: result.current.analyzing, results: result.current.results }; } }, {current:false}, {isFocused:()=>true}, result.current.restoreReview);
  let cleanup!: () => void;
  await act(async () => { cleanup = bind(); });
  expect(result.current.results?.detectedItems[0].itemType).toBe('jacket');
  expect(queue.getState().pendingUploads).toHaveLength(1);
  await act(async () => { await processor.processQueue(); });
  expect(ExternalAIService.classifyOnly).toHaveBeenCalledTimes(1);
  cleanup();
});
it('failed background analysis retains its entry without duplicate enqueue or modal alerts', async () => {
  queue.getState().addUpload(photo, 'image');
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({ success: false, classification: null, failureKind: 'network' });
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis()); processor.setAnalysisHook({ analyzeImage: result.current.analyzeImage });
  await act(async () => { await processor.processQueue(); });
  expect(queue.getState().pendingUploads).toHaveLength(1);
  expect(queue.getState().pendingUploads[0]).toMatchObject({ retryCount: 1, status: 'pending' });
  expect(alert).not.toHaveBeenCalled(); expect(result.current.results).toBeNull();
});
it('normal network failure flushes owner-tagged queue before acknowledging offline save', async () => {
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({ success: false, classification: null, failureKind: 'network' });
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { expect(await result.current.analyzeImage(photo)).toBe(false); });
  expect(queue.getState().pendingUploads[0]).toMatchObject({ ownerId: 'owner', uri: photo });
  expect(await AsyncStorage.getItem('wardrobe-upload-queue')).toContain(photo);
  expect(alert).toHaveBeenCalledWith('videoAnalysis.savedOffline', 'videoAnalysis.savedOfflinePhoto');
});
it('storage failure never acknowledges a durable offline save', async () => {
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({ success: false, classification: null, failureKind: 'network' });
  (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage(photo); });
  expect(alert).toHaveBeenCalledWith('videoAnalysis.analysisFailed', 'videoAnalysis.somethingWrong');
  expect(alert).not.toHaveBeenCalledWith('videoAnalysis.savedOffline', expect.anything());
});

it('native offline input is staged before the queue persists a durable URI', async () => {
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'ios' });
  (FileSystem.readAsStringAsync as jest.Mock).mockResolvedValueOnce('photo-base64');
  const stage = jest.spyOn(pendingMedia, 'stagePendingScanMedia').mockResolvedValueOnce('file:///documents/durable-scan.jpg');
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({ success: false, classification: null, failureKind: 'network' });
  jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage('file:///temporary/capture.jpg'); });
  expect(stage).toHaveBeenCalledWith('owner', 'file:///temporary/capture.jpg', 'image');
  expect(queue.getState().pendingUploads[0].uri).toBe('file:///documents/durable-scan.jpg');
  expect(await AsyncStorage.getItem('wardrobe-upload-queue')).not.toContain('file:///temporary/capture.jpg');
});

it('unmount after a committed copy but before queue acceptance removes only the unused managed copy', async () => {
  const owner='11111111-1111-4111-8111-111111111111';authEvents.emitLogin(owner);
  Object.defineProperty(Platform,'OS',{configurable:true,value:'ios'});
  (FileSystem.readAsStringAsync as jest.Mock).mockResolvedValueOnce('photo-base64');
  let finish!:(uri:string)=>void;
  const stage=jest.spyOn(pendingMedia,'stagePendingScanMedia').mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));
  const remove=jest.spyOn(pendingMedia,'removePendingScanMedia').mockResolvedValue(undefined);
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValue({success:false,classification:null,failureKind:'network'});
  const {result,unmount}=renderHook(()=>useVideoAnalysis());let request!:Promise<boolean>;
  act(()=>{request=result.current.analyzeImage('file:///camera-original.jpg');});
  await act(async()=>{for(let i=0;i<20;i++)await Promise.resolve();});expect(stage).toHaveBeenCalledTimes(1);
  unmount();
  const derived=`file:///documents/aiwardrobe-scan-queue/${owner}/44444444-4444-4444-8444-444444444444.jpg`;
  await act(async()=>{finish(derived);expect(await request).toBe(false);});
  expect(remove).toHaveBeenCalledWith(owner,derived,expect.any(Function));expect(queue.getState().pendingUploads).toEqual([]);
  expect(remove).not.toHaveBeenCalledWith(owner,'file:///camera-original.jpg');
});
