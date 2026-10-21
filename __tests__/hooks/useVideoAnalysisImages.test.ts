jest.mock('../../src/services/scanReviews',()=>({commitScanReview:jest.fn(async (_owner:any,result:any)=>result),findScanReviewBySource:jest.fn().mockResolvedValue(null)}));
jest.mock('../../src/services/videoFrameSampling', () => ({ readVideoSampleTimes: jest.fn().mockResolvedValue([0, 1000, 2000]), readAccurateVideoFrames:jest.fn() }));
import { act, renderHook } from '@testing-library/react-native';
import { Alert, Platform } from 'react-native';
import { readAccurateVideoFrames } from '../../src/services/videoFrameSampling';
import { useVideoAnalysis } from '../../src/features/wardrobe/useVideoAnalysis';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import { ExternalAIService } from '../../src/services/externalAIService';
jest.mock('../../lib/supabase', () => ({ supabase: { functions: { invoke: jest.fn() } } }));
jest.mock('expo-video-thumbnails', () => ({ getThumbnailAsync: jest.fn() }));
jest.mock('expo-file-system/legacy', () => ({ readAsStringAsync: jest.fn(), deleteAsync: jest.fn().mockResolvedValue(undefined) }));
jest.mock('../../src/services/externalAIService', () => ({
  ...jest.requireActual('../../src/services/externalAIService'),
  ExternalAIService: { classifyOnly: jest.fn(), processVideoFrames: jest.fn() },
}));
const descriptor = Object.getOwnPropertyDescriptor(Platform, 'OS')!;
beforeEach(() => {
  jest.clearAllMocks();
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'web' });
});
afterEach(() => { Object.defineProperty(Platform, 'OS', descriptor); jest.restoreAllMocks(); });

it.each(['unavailable', 'exception'])('reports failure without fabricated garments when analysis is %s', async mode => {
  const input = 'data:image/png;base64,iVBORw0KGgo=';
  const process = ExternalAIService.classifyOnly as jest.Mock;
  if (mode === 'exception') process.mockRejectedValueOnce(new Error('unavailable'));
  else process.mockResolvedValueOnce({ success: false, classification: null });
  jest.spyOn(console, 'error').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage(input); });
  expect(result.current.results).toBeNull();
  expect(result.current.analyzing).toBe(false);
});


it.each([
  ['tops', 'shirt', 'upper'], ['outerwear', 'jacket', 'upper'],
  ['bottoms', 'pants', 'lower'], ['dresses', 'dress', 'full'],
  ['shoes', 'boots', 'feet'], ['accessories', 'bag', 'accessory'],
])('preserves provider attributes and maps %s to the display position', async (section, category, position) => {
  const input = 'data:image/png;base64,iVBORw0KGgo=';
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValueOnce({
    success: true, imageUrl: input, cutoutUrl: null, steps: ['nvidia_classify'], description: 'Provider description',
    classification: { category, section, confidence: 0.91, attributes: { color: 'olive', material: 'cotton', style: 'casual' } },
  });
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage(input); });
  expect(result.current.results?.detectedItems).toEqual([expect.objectContaining({
    itemType: category, color: 'olive', material: 'cotton', position, confidence: 0.91, frameImage: input,
  })]);
  expect(result.current.results?.detectedItems[0].colorHex).toBeUndefined();
  expect(result.current.analyzing).toBe(false);
});

it('video with no extracted frames reports the problem without a fake manual detection', async () => {
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeVideo('file:///video.mp4'); });
  expect(result.current.results).toBeNull();
  expect(result.current.analyzing).toBe(false);
  expect(ExternalAIService.classifyOnly).not.toHaveBeenCalled();
  expect(alert).toHaveBeenCalled();
});

it.each([
  { section: 'outerwear', material: null, accepted: true },
  { section: 'invalid-section', material: 'cotton', accepted: false },
])('handles unknown attributes without inventing detected values: %j', async ({ section, material, accepted }) => {
  jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  (ExternalAIService.classifyOnly as jest.Mock).mockResolvedValueOnce({
    success: true, imageUrl: 'data:image/png;base64,iVBORw0KGgo=', steps: ['nvidia_classify'],
    classification: { category: 'jacket', section, confidence: 0.9, attributes: { color: 'beige', material, style: 'casual' } },
  });
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { await result.current.analyzeImage('data:image/png;base64,iVBORw0KGgo='); });
  if (accepted) expect(result.current.results?.detectedItems[0].material).toBeUndefined();
  else expect(result.current.results).toBeNull();
  expect(result.current.analyzing).toBe(false);
});

it('video sends all extracted frames once and preserves the later-frame garment image', async () => {
  Object.defineProperty(Platform, 'OS', { configurable: true, value: 'ios' });
  (VideoThumbnails.getThumbnailAsync as jest.Mock).mockImplementation(async (_uri, options) => ({ uri: `file:///frame-${options.time}.jpg` }));
  (FileSystem.readAsStringAsync as jest.Mock).mockImplementation(async uri => uri);
  (ExternalAIService.processVideoFrames as jest.Mock).mockImplementation(async frames => frames.map((frame: string, index: number) => ({
    success: true, imageUrl: frame, steps: ['nvidia_classify'], classification: {
      category: index === 0 ? 'shirt' : 'pants', section: index === 0 ? 'tops' : 'bottoms', confidence: 0,
      attributes: { color: 'beige', style: 'casual', material: null },
    },
  })));
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { expect(await result.current.analyzeVideo('file:///clip.mp4')).toBe(true); });
  expect(ExternalAIService.processVideoFrames).toHaveBeenCalledTimes(1);
  expect(ExternalAIService.processVideoFrames).toHaveBeenCalledWith(['file:///frame-0.jpg','file:///frame-1000.jpg','file:///frame-2000.jpg']);
  expect(result.current.results?.detectedItems[1]).toMatchObject({ itemType:'pants', frameImage:'file:///frame-1000.jpg', outfitId:2 });
  expect(ExternalAIService.classifyOnly).not.toHaveBeenCalled();
  expect(FileSystem.deleteAsync).toHaveBeenCalledTimes(3);
});
it('failed thumbnail reads still remove their temporary files and never call the provider', async () => {
  Object.defineProperty(Platform, 'OS', { configurable:true, value:'ios' });
  (VideoThumbnails.getThumbnailAsync as jest.Mock).mockImplementation(async (_uri, options) => ({uri:`file:///failed-${options.time}.jpg`}));
  (FileSystem.readAsStringAsync as jest.Mock).mockRejectedValue(new Error('read failed'));
  const { result } = renderHook(() => useVideoAnalysis());
  await act(async () => { expect(await result.current.analyzeVideo('file:///clip.mp4')).toBe(false); });
  expect(FileSystem.deleteAsync).toHaveBeenCalledTimes(3);
  expect(ExternalAIService.processVideoFrames).not.toHaveBeenCalled();
  expect(result.current.results).toBeNull();
});

it('Android uses accurate native frame images rather than the deprecated nearest-keyframe module', async () => {
  Object.defineProperty(Platform,'OS',{configurable:true,value:'android'});
  (readAccurateVideoFrames as jest.Mock).mockResolvedValue([{timeMs:3000,base64:'red'},{timeMs:15000,base64:'green'},{timeMs:27000,base64:'blue'}]);
  (ExternalAIService.processVideoFrames as jest.Mock).mockResolvedValue(['red','green','blue'].map(imageUrl=>({success:true,imageUrl,steps:[],classification:{category:'jacket',section:'outerwear',confidence:0,attributes:{color:'beige',style:'casual',material:null}}})));
  const {result}=renderHook(()=>useVideoAnalysis());
  await act(async()=>{expect(await result.current.analyzeVideo('file:///clip.mp4')).toBe(true);});
  expect(ExternalAIService.processVideoFrames).toHaveBeenCalledWith(['red','green','blue']);
  expect(VideoThumbnails.getThumbnailAsync).not.toHaveBeenCalled();
});

it('photo scan uses the real classify-only service request and preserves the original photo', async () => {
  const {supabase}=require('../../lib/supabase');
  const actual=jest.requireActual('../../src/services/externalAIService');
  const photo='data:image/jpeg;base64,cGhvdG8=';
  supabase.functions.invoke.mockResolvedValueOnce({error:null,data:{success:true,description:'Beige jacket',classification:{category:'jacket',section:'outerwear',confidence:0,attributes:{color:'beige',style:'casual',material:null}}}});
  (ExternalAIService.classifyOnly as jest.Mock).mockImplementationOnce(actual.ExternalAIService.classifyOnly);
  const {result}=renderHook(()=>useVideoAnalysis());let succeeded;
  await act(async()=>{succeeded=await result.current.analyzeImage(photo);});
  expect(succeeded).toBe(true);
  expect(supabase.functions.invoke).toHaveBeenCalledWith('ai-process',{body:{image:photo,operation:'classify'},signal:expect.anything()});
  expect(supabase.functions.invoke).toHaveBeenCalledTimes(1);
  expect(result.current.results?.detectedItems[0]).toMatchObject({itemType:'jacket',frameImage:photo,detectionSources:['nvidia_classify']});
  expect(result.current.results?.detectedItems[0].cutoutImage).toBeUndefined();
});
