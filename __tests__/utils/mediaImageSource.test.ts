import { mediaImageSource, nativeMediaImageSource, prefetchMediaImages } from '../../src/utils/mediaImageSource';
jest.mock('../../lib/supabase', () => ({ supabase: {} }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
jest.mock('expo-file-system/legacy', () => ({ documentDirectory: 'file:///current/Documents/' }));
const owner = '11111111-1111-4111-8111-111111111111';
const uri = `https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/image/original.jpg`;
const identity = { userId: owner, accessToken: 'current-session' };

test('private sources preserve metadata and replace stale credentials with current owner token', () => {
  const result = mediaImageSource({ uri, width: 400, headers: { Authorization: 'old', Accept: 'image/*' } }, identity) as Record<string, unknown>;
  expect(result).toMatchObject({ uri, width: 400, headers: { Authorization: 'Bearer current-session', Accept: 'image/*' } });
  expect(result.cacheKey).not.toContain('session');
});
test('signed imports render through canonical authentication, never the signed cache', () => {
  const signed = uri.replace('/authenticated/','/sign/')+'?token=old';
  expect(mediaImageSource(signed, identity)).toMatchObject({ uri, headers: { Authorization: 'Bearer current-session' } });
});
it.each([{}, { userId: 'other', accessToken: 'current-session' }, { userId: owner }])('blocks private image/cache access without current owner authorization %j', unavailable => {
  expect(mediaImageSource({ uri, headers: { Authorization: 'old' } }, unavailable)).toBeUndefined();
});
test('static, inline and external sources retain their original shape without credentials', () => {
  for (const input of [42, 'data:image/png;base64,AAAA', 'https://catalog.invalid/image.jpg', { uri: 'file:///photo.jpg', width: 300 }]) {
    expect(mediaImageSource(input, identity)).toBe(input);
  }
});
test('mixed source arrays cannot carry stale private entries into another account', () => {
  const external = { uri: 'https://catalog.invalid/image.jpg' };
  expect(mediaImageSource([{ uri }, external], {})).toEqual([external]);
  expect(mediaImageSource([{ uri }], {})).toBeUndefined();
});
test('mixed prefetch keeps credentials exclusively on owned private URLs', async () => {
  const prefetch = jest.fn().mockResolvedValue(true), external = 'https://catalog.invalid/image.jpg';
  expect(await prefetchMediaImages([uri, external], 'memory-disk', identity, prefetch)).toBe(true);
  expect(prefetch.mock.calls).toEqual([
    [uri, { cachePolicy: 'memory-disk', headers: { Authorization: 'Bearer current-session' } }],
    [external, 'memory-disk'],
  ]);
  prefetch.mockClear();
  expect(await prefetchMediaImages([uri], undefined, {}, prefetch)).toBe(false);
  expect(prefetch).not.toHaveBeenCalled();
});

test('managed local photos rebase to Documents and remain owner isolated', () => {
  const uri = `file:///old/Documents/aiwardrobe-pending-media/${owner}/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa.jpg`;
  const result = mediaImageSource({ uri, headers: { Authorization: 'old' } }, identity) as {uri:string;headers?:unknown};
  expect(result.uri).toContain('file:///current/Documents/');
  expect(result.headers).toBeUndefined();
  expect(mediaImageSource({ uri }, {})).toBeUndefined();
  expect(mediaImageSource({ uri }, { userId: 'other', accessToken: 'other' })).toBeUndefined();
});


test('authorized Android object sources use the header-forwarding array form',()=>{
 const input={uri,headers:{Authorization:'Bearer fixture'},width:40};
 expect(nativeMediaImageSource(input,'android',{width:80,height:60})).toEqual([{...input,width:40,height:60}]);
 expect(input).toEqual({uri,headers:{Authorization:'Bearer fixture'},width:40});
});
test('iOS sources, existing arrays and sources without headers retain their form',()=>{
 const input={uri,headers:{Authorization:'Bearer fixture'}},array=[input],publicSource={uri:'https://public.test/photo'};
 expect(nativeMediaImageSource(input,'ios')).toBe(input);expect(nativeMediaImageSource(array,'android')).toBe(array);
 expect(nativeMediaImageSource(publicSource,'android')).toBe(publicSource);expect(nativeMediaImageSource(3,'android')).toBe(3);
});

test('Android never shares private candidate headers with an unauthenticated candidate',()=>{
 const privateSource=mediaImageSource({uri},identity);
 const mixed=[privateSource,{uri:'https://public.test/photo'}];
 expect(nativeMediaImageSource(mixed,'android')).toBeUndefined();
 const sameScope=[privateSource,mediaImageSource({uri:uri.replace('original.jpg','larger.jpg')},identity)];
 expect(nativeMediaImageSource(sameScope,'android')).toBe(sameScope);
});
