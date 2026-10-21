import { processPendingActions, type PendingAction } from '../../store/wardrobeSyncService';
import { prepareWardrobeItemMedia } from '../../src/services/wardrobeItemMedia';
import { authEvents } from '../../store/authEvents';
import type { ClothingItem } from '../../src/types/domain';
const mockUpsert=jest.fn(), mockFrom=jest.fn();
jest.mock('../../src/services/wardrobeItemMedia',()=>({prepareWardrobeItemMedia:jest.fn()}));
jest.mock('../../lib/supabase',()=>({supabase:{auth:{getSession:async()=>({data:{session:{user:{id:'owner'}}}})},from:(...args:unknown[])=>mockFrom(...args)}}));
const item={id:'garment',imageUrl:'data:image/jpeg;base64,/9j/2Q==',updatedAt:'2026-10-17T00:00:00Z'} as ClothingItem;
const action:PendingAction={id:'add',type:'add_item',payload:item as unknown as Record<string,unknown>,createdAt:''};
beforeEach(()=>{
  jest.clearAllMocks();authEvents.emitLogin('owner');
  mockFrom.mockReturnValue({select:()=>({in:async()=>({data:[],error:null})}),upsert:mockUpsert});
  mockUpsert.mockResolvedValue({error:null});
  (prepareWardrobeItemMedia as jest.Mock).mockResolvedValue({...item,imageUrl:'https://storage.test/original.jpg'});
});
test('record writes contain confirmed media references and return them for persistence',async()=>{
  const result=await processPendingActions([action],[item],'owner');
  expect(prepareWardrobeItemMedia).toHaveBeenCalledWith('owner',item);
  expect(mockUpsert).toHaveBeenCalledWith([expect.objectContaining({image_url:'https://storage.test/original.jpg'})]);
  expect(result).toMatchObject({processedIds:['add'],updatedItems:[{imageUrl:'https://storage.test/original.jpg'}]});
});
test('media failure retains original work and blocks record insertion and dependent wear logs',async()=>{
  (prepareWardrobeItemMedia as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
  try {
    const result=await processPendingActions([action,{id:'wear',type:'add_wear_log',payload:{id:'log',itemIds:['garment']},createdAt:''}],[item],'owner');
    expect(result).toEqual({processedIds:[],updatedItems:[item]});
    expect(mockUpsert).not.toHaveBeenCalled();
  } finally {error.mockRestore();}
});
test('record failure does not publish uploaded references or acknowledge work',async()=>{
  mockUpsert.mockResolvedValueOnce({error:new Error('offline')});
  const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
  try {expect(await processPendingActions([action],[item],'owner')).toEqual({processedIds:[],updatedItems:[item]});}
  finally {error.mockRestore();}
});

test('server capacity rejection preserves original media and identifies only the blocked actions',async()=>{
 mockUpsert.mockResolvedValueOnce({error:{code:'P0001',message:'WARDROBE_LIMIT_REACHED',details:'{"limit":20}'}});
 const log=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{
  const result=await processPendingActions([action],[item],'owner');
  expect(result).toMatchObject({processedIds:[],updatedItems:[item],capacityIssue:{limit:20,actionIds:['add']}});
 }finally{log.mockRestore();}
});
test.each([
 {code:'42501',message:'WARDROBE_LIMIT_REACHED',details:'{"limit":20}'},
 {code:'P0001',message:'other failure',details:'{"limit":20}'},
 {code:'P0001',message:'WARDROBE_LIMIT_REACHED',details:'{"limit":-1}'},
 {code:'P0001',message:'WARDROBE_LIMIT_REACHED',details:'invalid JSON'},
])('unrecognized server errors are not presented as capacity decisions (%j)',async error=>{
 mockUpsert.mockResolvedValueOnce({error});const log=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{expect((await processPendingActions([action],[item],'owner')).capacityIssue).toBeUndefined();}finally{log.mockRestore();}
});
