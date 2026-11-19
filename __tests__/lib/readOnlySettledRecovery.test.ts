/** @jest-environment node */
import {pollSettledReadOnlyResult} from '../../src/lib/readOnlySettledRecovery';

describe('already-paid Stylist 6 result recovery',()=>{
  test('signed PHOTO result stored after the first unsuccessful SELECT is recovered without a second generation',async()=>{
    const approved={success:true,generationRequestId:'same-owner-request',
      visualStatus:'NVIDIA_APPLIED',outfits:[{id:'approved-real-photo'}]};
    const read=jest.fn().mockResolvedValueOnce(null)
      .mockResolvedValueOnce(null).mockResolvedValueOnce(approved);
    const answer=await pollSettledReadOnlyResult(read,{
      maxWaitMs:600,intervalMs:10,queryTimeoutMs:100});
    expect(answer).toBe(approved);
    expect(read).toHaveBeenCalledTimes(3);
    expect(read.mock.calls.every(([signal])=>signal instanceof AbortSignal)).toBe(true);
  });

  test('a server that has NOT completed never fabricates a paid deck',async()=>{
    const read=jest.fn().mockResolvedValue(null);
    expect(await pollSettledReadOnlyResult(read,{
      maxWaitMs:45,intervalMs:10,queryTimeoutMs:20})).toBeNull();
    expect(read.mock.calls.length).toBeGreaterThan(1);
  });

  test('a cancelled native screen stops pending read-only polls immediately',async()=>{
    const cancel=new AbortController();
    const read=jest.fn().mockResolvedValue(null);
    const pending=pollSettledReadOnlyResult(read,{
      signal:cancel.signal,maxWaitMs:1000,intervalMs:400});
    setTimeout(()=>cancel.abort(),15);
    expect(await pending).toBeNull();
    expect(read).toHaveBeenCalledTimes(1);
    expect(await pollSettledReadOnlyResult(read,{
      signal:cancel.signal,maxWaitMs:1000})).toBeNull();
    expect(read).toHaveBeenCalledTimes(1);
  });

  test('one hung RLS fetch cannot block recovery forever',async()=>{
    const read=jest.fn((_signal:AbortSignal)=>new Promise<null>(()=>{}));
    const started=Date.now();
    expect(await pollSettledReadOnlyResult(read,{
      maxWaitMs:65,intervalMs:10,queryTimeoutMs:20})).toBeNull();
    expect(Date.now()-started).toBeLessThan(350);
  });

  test('transient RLS error can recover only by later successful READ',async()=>{
    const correct={success:true,outfits:[{id:'one-real-source-look'}]};
    const read=jest.fn().mockRejectedValueOnce(new Error('network temporarily lost'))
      .mockResolvedValueOnce(correct);
    expect(await pollSettledReadOnlyResult(read,{
      maxWaitMs:160,intervalMs:10,queryTimeoutMs:40})).toBe(correct);
    expect(read).toHaveBeenCalledTimes(2);
  });
});
