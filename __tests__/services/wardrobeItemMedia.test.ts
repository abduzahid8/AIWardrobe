import { prepareWardrobeItemMedia } from '../../src/services/wardrobeItemMedia';
import { uploadWardrobeImage } from '../../src/services/wardrobeMedia';
import type { ClothingItem } from '../../src/types/domain';
jest.mock('../../src/services/wardrobeMedia', () => ({
  ...jest.requireActual('../../src/services/wardrobeMedia'), uploadWardrobeImage: jest.fn(),
}));
jest.mock('../../lib/supabase', () => ({ supabase: {} }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
jest.mock('expo-file-system/legacy', () => ({ documentDirectory: 'file:///restored/Documents/' }));
const owner='11111111-1111-4111-8111-111111111111', id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const canonical=`https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/${id}/original.jpg`;
const item=(imageUrl:string,thumbnailUrl?:string)=>({id,imageUrl,thumbnailUrl,name:'shirt'} as ClothingItem);
beforeEach(()=>{jest.clearAllMocks();(uploadWardrobeImage as jest.Mock).mockResolvedValue(canonical);});
test('remote catalog sources require no download or upload',async()=>{
  const input=item('https://catalog.test/photo.jpg');
  expect(await prepareWardrobeItemMedia(owner,input)).toEqual(input);
  expect(uploadWardrobeImage).not.toHaveBeenCalled();
});
test('signed references become permanent owner-scoped references',async()=>{
  expect((await prepareWardrobeItemMedia(owner,item(canonical.replace('/authenticated/','/sign/')+'?token=secret'))).imageUrl).toBe(canonical);
  await expect(prepareWardrobeItemMedia('22222222-2222-4222-8222-222222222222',item(canonical))).rejects.toMatchObject({code:'access_denied'});
  expect(uploadWardrobeImage).not.toHaveBeenCalled();
});
test('managed device photo rebases and original/thumbnail use separate object names',async()=>{
  const old=`file:///old/Documents/aiwardrobe-pending-media/${owner}/${id}.jpg`;
  await prepareWardrobeItemMedia(owner,item(old,'data:image/jpeg;base64,/9j/2Q=='));
  expect(uploadWardrobeImage).toHaveBeenNthCalledWith(1,owner,id,old.replace('/old/','/restored/'),undefined,'original');
  expect(uploadWardrobeImage).toHaveBeenNthCalledWith(2,owner,id,'data:image/jpeg;base64,/9j/2Q==',undefined,'thumbnail');
});
test('identical thumbnail reuses original and never uploads twice',async()=>{
  const source='data:image/jpeg;base64,/9j/2Q==';
  expect(await prepareWardrobeItemMedia(owner,item(source,source))).toMatchObject({imageUrl:canonical,thumbnailUrl:canonical});
  expect(uploadWardrobeImage).toHaveBeenCalledTimes(1);
});
test('failed media upload cannot return a prepared record',async()=>{
  (uploadWardrobeImage as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  await expect(prepareWardrobeItemMedia(owner,item('file:///cache/photo.jpg'))).rejects.toThrow('offline');
});
