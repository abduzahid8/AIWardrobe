import {createHash} from 'crypto';
import * as Crypto from 'expo-crypto';
import {hashAuditBytes} from '../../scripts/validation/native-media-digest';

jest.mock('expo-crypto',()=>({
 CryptoDigestAlgorithm:{SHA256:'SHA-256'},
 digest:jest.fn(async (_algorithm:string, bytes:unknown)=>{
  if(!(bytes instanceof Uint8Array))throw new TypeError('Native Android requires TypedArray');
  const output=require('crypto').createHash('sha256').update(bytes).digest();
  return Uint8Array.from(output).buffer;
 }),
}));
beforeEach(()=>jest.clearAllMocks());

test('known SHA-256 passes bytes through the Android TypedArray boundary',async()=>{
 expect(await hashAuditBytes(new Uint8Array([97,98,99]).buffer)).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
 expect((Crypto.digest as jest.Mock).mock.calls[0][1]).toBeInstanceOf(Uint8Array);
});

test('full-sized diagnostic input matches an independent reference hash',async()=>{
 const bytes=new Uint8Array(788309);
 expect(await hashAuditBytes(bytes.buffer)).toBe(createHash('sha256').update(bytes).digest('hex'));
 expect((Crypto.digest as jest.Mock).mock.calls[0][1].byteLength).toBe(788309);
});

test('native digest failure rejects instead of producing an integrity result',async()=>{
 (Crypto.digest as jest.Mock).mockRejectedValueOnce(new Error('Native unavailable'));
 await expect(hashAuditBytes(new Uint8Array([1]).buffer)).rejects.toThrow('Native unavailable');
});
