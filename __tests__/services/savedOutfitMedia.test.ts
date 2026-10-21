import { saveCloudOutfit } from '../../src/services/savedOutfitService';
import { prepareWardrobeItemMedia } from '../../src/services/wardrobeItemMedia';
import { authEvents } from '../../store/authEvents';
const mockUpsert=jest.fn(), mockSession=jest.fn();
jest.mock('../../lib/supabase',()=>({supabase:{auth:{getSession:()=>mockSession()},from:()=>({upsert:mockUpsert})}}));
jest.mock('../../src/services/wardrobeItemMedia',()=>({prepareWardrobeItemMedia:jest.fn()}));
const input={id:'outfit',date:'2026-10-17',occasion:'casual',items:[{id:'garment',image:'file:///photo.jpg',imageUrl:'file:///thumb.jpg',name:'shirt'}]};
beforeEach(()=>{
 jest.clearAllMocks();authEvents.emitLogin('owner');mockSession.mockResolvedValue({data:{session:{user:{id:'owner'}}},error:null});
 mockUpsert.mockReturnValue({select:()=>({single:async()=>({data:{id:'outfit'},error:null})})});
 (prepareWardrobeItemMedia as jest.Mock).mockResolvedValue({imageUrl:'https://storage.test/original.jpg',thumbnailUrl:'https://storage.test/thumbnail.jpg'});
});
test('both image field conventions persist permanent prepared references without mutating caller input',async()=>{
 const confirmed = await saveCloudOutfit(input);
 expect(confirmed).toEqual([{id:'garment',name:'shirt',image:'https://storage.test/original.jpg',imageUrl:'https://storage.test/thumbnail.jpg'}]);
 expect(mockUpsert).toHaveBeenCalledWith(expect.objectContaining({items:[{id:'garment',name:'shirt',image:'https://storage.test/original.jpg',imageUrl:'https://storage.test/thumbnail.jpg'}]}));
 expect(input.items[0].image).toBe('file:///photo.jpg');
});
it.each(['image','imageUrl'] as const)('single %s field is preserved without adding the other convention',async(field)=>{
 await saveCloudOutfit({...input,items:[{id:'garment',[field]:'file:///photo.jpg'}]});
 expect(mockUpsert.mock.calls[0][0].items).toEqual([{id:'garment',[field]:'https://storage.test/original.jpg'}]);
});
test('snapshot capture prevents caller changes during session lookup changing the saved look',async()=>{
 const value={...input,items:input.items.map(item=>({...item}))};
 mockSession.mockImplementationOnce(async()=>{value.items[0].image='file:///changed.jpg';value.items[0].name='changed';return {data:{session:{user:{id:'owner'}}}};});
 await saveCloudOutfit(value);
 expect(prepareWardrobeItemMedia).toHaveBeenCalledWith('owner',expect.objectContaining({imageUrl:'file:///photo.jpg'}));
 expect(mockUpsert.mock.calls[0][0].items[0].name).toBe('shirt');
});
test('media failure leaves saved outfit record unwritten',async()=>{
 (prepareWardrobeItemMedia as jest.Mock).mockRejectedValueOnce(new Error('upload failed'));
 await expect(saveCloudOutfit(input)).rejects.toThrow('upload failed');expect(mockUpsert).not.toHaveBeenCalled();
});
test('account change during media preparation prevents outfit write',async()=>{
 (prepareWardrobeItemMedia as jest.Mock).mockImplementationOnce(async()=>{authEvents.emitLogout();authEvents.emitLogin('other');return {imageUrl:'https://storage.test/original.jpg'};});
 await expect(saveCloudOutfit(input)).rejects.toThrow('Account changed');expect(mockUpsert).not.toHaveBeenCalled();
});
