import { createStore } from 'zustand/vanilla';
import { createItemsSlice } from '../../store/slices/wardrobeItemsSlice';
import { createNativeWardrobeThumbnail } from '../../src/services/wardrobeThumbnail';
import { stageNativeWardrobeImage, removePendingWardrobeImage } from '../../src/services/pendingWardrobeMedia';
import { authEvents } from '../../store/authEvents';
import type { WardrobeState } from '../../store/wardrobeStore';
const owner='11111111-1111-4111-8111-111111111111';
let mockOwner: string | undefined=owner;
let mockCap=-1;
jest.mock('../../store/auth',()=>({__esModule:true,default:{getState:()=>({user:mockOwner?{id:mockOwner}:null})}}));
jest.mock('../../store/subscriptionStore',()=>({__esModule:true,default:{getState:()=>({tier:'free',effectiveTier:'free',refreshTimeState:()=>{}})},TIER_FEATURES:{free:{get wardrobeItems(){return mockCap;}}}}));
jest.mock('../../src/services/pendingWardrobeMedia',()=>({stageNativeWardrobeImage:jest.fn(),removePendingWardrobeImage:jest.fn().mockResolvedValue(undefined)}));
jest.mock('../../src/services/wardrobeThumbnail',()=>({createNativeWardrobeThumbnail:jest.fn()}));
jest.mock('../../lib/supabase',()=>({supabase:{}}));
function fixture(){return createStore<WardrobeState>()((set,get,store)=>({...createItemsSlice(set,get,store),pendingActions:[],outfits:[],syncToServer:jest.fn().mockResolvedValue(undefined)} as unknown as WardrobeState));}
const input={category:'top',imageUrl:'file:///picker/photo.jpg',userId:'stale-input-owner'} as never;
beforeEach(()=>{jest.clearAllMocks();(createNativeWardrobeThumbnail as jest.Mock).mockImplementation(async(o:string,id:string)=>`file:///Documents/aiwardrobe-pending-media/${o}/${id}.png`);mockOwner=owner;mockCap=-1;authEvents.emitLogin(owner);(stageNativeWardrobeImage as jest.Mock).mockImplementation(async(o:string,id:string)=>`file:///Documents/aiwardrobe-pending-media/${o}/${id}.jpg`);});
test('initial save stages bytes before publishing the item or queue',async()=>{
 const state=fixture();let staged!:(uri:string)=>void;
 (stageNativeWardrobeImage as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{staged=resolve;}));
 const pending=state.getState().addItem(input);
 expect(state.getState().items).toEqual([]);expect(state.getState().pendingActions).toEqual([]);expect(state.getState().syncToServer).not.toHaveBeenCalled();
 const id=(stageNativeWardrobeImage as jest.Mock).mock.calls[0][1];const uri=`file:///Documents/aiwardrobe-pending-media/${owner}/${id}.jpg`;
 staged(uri);await pending;
 expect(state.getState().items[0]).toMatchObject({id,userId:owner,imageUrl:uri});
 expect(state.getState().pendingActions[0].payload).toMatchObject({id,imageUrl:uri});expect(state.getState().syncToServer).toHaveBeenCalledTimes(1);
});
test('file failure does not create an item or upload action',async()=>{
 const state=fixture();(stageNativeWardrobeImage as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(state.getState().addItem(input)).rejects.toThrow('disk full');expect(state.getState().items).toEqual([]);expect(state.getState().pendingActions).toEqual([]);expect(state.getState().syncToServer).not.toHaveBeenCalled();
});
test('account switch during staging cannot publish previous-account work',async()=>{
 const state=fixture();(stageNativeWardrobeImage as jest.Mock).mockImplementationOnce(async()=>{authEvents.emitLogout();mockOwner='22222222-2222-4222-8222-222222222222';authEvents.emitLogin(mockOwner);return 'file:///old.jpg';});
 await expect(state.getState().addItem(input)).rejects.toMatchObject({code:'account_changed'});expect(state.getState().items).toEqual([]);expect(state.getState().pendingActions).toEqual([]);
});
test('distinct supplied thumbnail is staged; identical original source generates a compact image',async()=>{
 const state=fixture();await state.getState().addItem({...input as object,thumbnailUrl:'file:///picker/thumb.jpg'} as never);
 const calls=(stageNativeWardrobeImage as jest.Mock).mock.calls;expect(calls).toHaveLength(2);expect(calls[0][1]).not.toBe(calls[1][1]);expect(createNativeWardrobeThumbnail).not.toHaveBeenCalled();
 jest.clearAllMocks();await state.getState().addItem({...input as object,thumbnailUrl:'file:///picker/photo.jpg'} as never);expect(stageNativeWardrobeImage).toHaveBeenCalledTimes(1);expect(createNativeWardrobeThumbnail).toHaveBeenCalledTimes(1);
});
test('cap is rechecked after staging to prevent concurrent saves exceeding it',async()=>{
 const state=fixture();mockCap=1;let finish!:(uri:string)=>void;
 (stageNativeWardrobeImage as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{finish=resolve;}));
 const pending=state.getState().addItem(input);await state.getState().addItem({category:'top',imageUrl:'https://catalog.test/photo.jpg'} as never);finish('file:///Documents/photo.jpg');
 await expect(pending).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED'});expect(state.getState().items).toHaveLength(1);expect(state.getState().pendingActions).toHaveLength(1);
});
test('native local photos require an authenticated owner before staging',async()=>{
 mockOwner=undefined;const state=fixture();await expect(state.getState().addItem(input)).rejects.toMatchObject({code:'access_denied'});expect(stageNativeWardrobeImage).not.toHaveBeenCalled();
});


test('thumbnail generation failure cannot publish an original-only item or upload action',async()=>{
 const state=fixture();(createNativeWardrobeThumbnail as jest.Mock).mockRejectedValueOnce(new Error('encode failed'));
 await expect(state.getState().addItem(input)).rejects.toThrow('encode failed');
 expect(state.getState().items).toEqual([]);expect(state.getState().pendingActions).toEqual([]);expect(state.getState().syncToServer).not.toHaveBeenCalled();
 expect(removePendingWardrobeImage).toHaveBeenCalledWith(owner,expect.stringMatching(/\.jpg$/));
});

test('supplied preview staging rejection rolls back only the newly staged original',async()=>{
 const state=fixture();
 // The first call succeeds; the second fails before returning a durable preview.
 (stageNativeWardrobeImage as jest.Mock).mockReset().mockImplementationOnce(async(o:string,id:string)=>`file:///Documents/aiwardrobe-pending-media/${o}/${id}.jpg`).mockRejectedValueOnce(new Error('preview disk failure'));
 await expect(state.getState().addItem({...input as object,thumbnailUrl:'file:///picker/preview.png'} as never)).rejects.toThrow('preview disk failure');
 expect(removePendingWardrobeImage).toHaveBeenCalledTimes(1);
 expect(removePendingWardrobeImage).toHaveBeenCalledWith(owner,expect.stringMatching(/\.jpg$/));
 expect(state.getState().pendingActions).toEqual([]);
});

test('limit reached after both staged files removes both without touching the item that occupied the slot',async()=>{
 const state=fixture();mockCap=1;
 (createNativeWardrobeThumbnail as jest.Mock).mockImplementationOnce(async(o:string,id:string)=>{
  await state.getState().addItem({category:'top',imageUrl:'https://catalog.test/retained.jpg'} as never);
  return `file:///Documents/aiwardrobe-pending-media/${o}/${id}.png`;
 });
 await expect(state.getState().addItem(input)).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED'});
 expect(removePendingWardrobeImage).toHaveBeenCalledTimes(2);
 expect(state.getState().items).toHaveLength(1);expect(state.getState().items[0].imageUrl).toBe('https://catalog.test/retained.jpg');
});

test('sync rejection after queue publication retains staged files for retry',async()=>{
 const state=fixture();(state.getState().syncToServer as jest.Mock).mockRejectedValueOnce(new Error('offline'));
 await expect(state.getState().addItem(input)).rejects.toThrow('offline');
 expect(state.getState().pendingActions).toHaveLength(1);expect(state.getState().items).toHaveLength(1);
 expect(removePendingWardrobeImage).not.toHaveBeenCalled();
});


test('rollback attempts every file and reports deletion failure without publishing',async()=>{
 const state=fixture();mockCap=1;
 (createNativeWardrobeThumbnail as jest.Mock).mockImplementationOnce(async(o:string,id:string)=>{
  await state.getState().addItem({category:'top',imageUrl:'https://catalog.test/retained.jpg'} as never);
  return `file:///Documents/aiwardrobe-pending-media/${o}/${id}.png`;
 });
 (removePendingWardrobeImage as jest.Mock).mockRejectedValueOnce(new Error('synthetic deletion rejection'));
 await expect(state.getState().addItem(input)).rejects.toMatchObject({code:'photo_cleanup_failed',cause:{code:'WARDROBE_LIMIT_REACHED'}});
 expect(removePendingWardrobeImage).toHaveBeenCalledTimes(2);
 expect(state.getState().items).toHaveLength(1);expect(state.getState().pendingActions).toHaveLength(1);
});

test('a stable review save retries the same item rather than duplicating it at the wardrobe cap',async()=>{
 const state=fixture(),stable='44444444-4444-4444-8444-444444444444';mockCap=1;
 const owned={...(input as object),userId:owner} as never;
 await state.getState().addItem(owned,{idempotencyId:stable});
 await state.getState().addItem(owned,{idempotencyId:stable});
 expect(state.getState().items).toHaveLength(1);expect(state.getState().pendingActions).toHaveLength(1);expect(state.getState().items[0].id).toBe(stable);
});
test('overlapping creates for one review item share one media staging and mutation',async()=>{
 const state=fixture(),stable='55555555-5555-4555-8555-555555555555';let finish!:(uri:string)=>void;
 (stageNativeWardrobeImage as jest.Mock).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
 const owned={...(input as object),userId:owner} as never;
 const first=state.getState().addItem(owned,{idempotencyId:stable}),second=state.getState().addItem(owned,{idempotencyId:stable});
 expect(stageNativeWardrobeImage).toHaveBeenCalledTimes(1);finish(`file:///Documents/aiwardrobe-pending-media/${owner}/${stable}.jpg`);
 await Promise.all([first,second]);expect(state.getState().items).toHaveLength(1);expect(state.getState().pendingActions).toHaveLength(1);
});

test('a server capacity rejection is reported as retained local work, including stable review retries',async()=>{
 const state=fixture(),stable='66666666-6666-4666-8666-666666666666';
 (state.getState().syncToServer as jest.Mock).mockImplementation(async()=>{
  state.setState({wardrobeCapacityIssue:{limit:20,actionIds:state.getState().pendingActions.map(a=>a.id)}});
 });
 const owned={...(input as object),userId:owner} as never;
 await expect(state.getState().addItem(owned,{idempotencyId:stable})).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED',savedLocally:true,limit:20});
 await expect(state.getState().addItem(owned,{idempotencyId:stable})).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED',savedLocally:true});
 expect(state.getState().items).toHaveLength(1);expect(state.getState().pendingActions).toHaveLength(1);
 expect(removePendingWardrobeImage).not.toHaveBeenCalled();
});
