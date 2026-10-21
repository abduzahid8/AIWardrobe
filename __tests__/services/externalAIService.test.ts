import { ExternalAIService } from '../../src/services/externalAIService';
import { supabase } from '../../lib/supabase';

jest.mock('../../lib/supabase', () => ({
  supabase: {
    functions: {
      invoke: jest.fn(),
    },
  },
}));

const mockedInvoke = supabase.functions.invoke as jest.Mock;

describe('ExternalAIService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses the studio_photo operation and returns the cutout image', async () => {
    mockedInvoke.mockResolvedValue({
      data: {
        success: true,
        cutoutUrl: 'https://cdn.example.com/cutout.png',
        classification: {
          category: 'shirt',
          section: 'tops',
          confidence: 0.91,
          attributes: {
            style: 'casual',
            color: 'blue',
            material: 'cotton',
          },
        },
        description: 'Blue cotton shirt',
      },
      error: null,
    });

    const result = await ExternalAIService.processStudioPhoto('abc123');

    expect(mockedInvoke).toHaveBeenCalledWith('ai-process', {
      signal: expect.anything(),
      body: {
        image: 'abc123',
        operation: 'classify',
      },
    });
    expect(result.success).toBe(true);
    expect(result.imageUrl).toBe('data:image/jpeg;base64,abc123');
    expect(result.cutoutUrl).toBe('https://cdn.example.com/cutout.png');
    expect(result.steps).toEqual(['nvidia_classify']);
  });

  it('falls back to the original image when studio processing fails', async () => {
    mockedInvoke.mockResolvedValue({
      data: {
        success: false,
        error: 'processing failed',
      },
      error: null,
    });

    const result = await ExternalAIService.processStudioPhoto('abc123');

    expect(result.success).toBe(false);
    expect(result.imageUrl).toBe('data:image/jpeg;base64,abc123');
    expect(result.steps).toEqual(['error_fallback']);
  });
  it.each([
    ['FunctionsFetchError', 'fetch failed', 'network'],
    ['TimeoutError', 'request expired', 'timeout'],
    ['Error', 'provider rejected request', 'provider'],
  ])('classifies %s failures for offline retry without exposing provider bodies', async (name, message, expected) => {
    mockedInvoke.mockResolvedValue({ data: null, error: Object.assign(new Error(message), { name }) });
    const result = await ExternalAIService.classifyOnly('photo');
    expect(result.success).toBe(false);
    expect(result.failureKind).toBe(expected);
  });

});

describe('video frame batch contract', () => {
  const classification = { category: 'jacket', section: 'outerwear', confidence: 0, attributes: { style: 'casual', color: 'beige', material: null } };
  it('sends all frames once and binds each returned index to its original image', async () => {
    const invoke = supabase.functions.invoke as jest.Mock;
    invoke.mockResolvedValueOnce({ data: { success: true, frames: [0,1].map(frameIndex => ({ frameIndex, classification })) }, error: null });
    const results = await ExternalAIService.processVideoFrames(['first', 'second']);
    expect(invoke).toHaveBeenLastCalledWith('ai-process', { signal: expect.anything(), body: { operation: 'classify_frames', frames: ['first', 'second'] } });
    expect(results.map(result => result.imageUrl)).toEqual(['data:image/jpeg;base64,first', 'data:image/jpeg;base64,second']);
  });
  it.each([[], [{ frameIndex: 1, classification }], [{ frameIndex: 0 }]].map(frames => [frames]))('rejects incomplete or misbound batches %j', async frames => {
    (supabase.functions.invoke as jest.Mock).mockResolvedValueOnce({ data: { success: true, frames }, error: null });
    await expect(ExternalAIService.processVideoFrames(['first'])).rejects.toThrow();
  });
});

describe('complete scan request deadline', () => {
 beforeEach(() => { jest.useFakeTimers(); mockedInvoke.mockReset(); });
 afterEach(() => { jest.useRealTimers(); });
 it('ends a stalled photo call without publishing a classification, even if transport ignores abort', async () => {
  let complete: (value:any)=>void;
  mockedInvoke.mockImplementationOnce(() => new Promise(resolve => { complete=resolve; }));
  const pending=ExternalAIService.classifyOnly('original');
  await Promise.resolve();const signal=mockedInvoke.mock.calls[0][1].signal;
  await jest.advanceTimersByTimeAsync(60_000);
  const result=await pending;
  expect(result).toMatchObject({success:false,failureKind:'timeout',classification:null,imageUrl:'data:image/jpeg;base64,original'});
  expect(signal.aborted).toBe(true);
  complete!({data:{success:true,classification:{category:'jacket'}},error:null});await Promise.resolve();
  expect(result.success).toBe(false);expect(jest.getTimerCount()).toBe(0);
 });
 it.each([1,3])('bounds a stalled %i-frame video batch by its sequential server budget', async count => {
  mockedInvoke.mockImplementationOnce(() => new Promise(()=>{}));
  const pending=ExternalAIService.processVideoFrames(Array(count).fill('frame'));
  const rejection=expect(pending).rejects.toMatchObject({name:'TimeoutError'});
  await Promise.resolve();await jest.advanceTimersByTimeAsync(count*45_000+15_000);
  await rejection;expect(mockedInvoke.mock.calls[0][1].signal.aborted).toBe(true);
  expect(jest.getTimerCount()).toBe(0);
 });
 it('clears the deadline after a quick provider failure', async () => {
  mockedInvoke.mockResolvedValueOnce({data:null,error:new Error('provider rejected')});
  await ExternalAIService.classifyOnly('original');expect(jest.getTimerCount()).toBe(0);
 });
});
