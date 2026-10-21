import { createStore } from 'zustand/vanilla';
import { createItemsSlice } from '../../store/slices/wardrobeItemsSlice';
import useSubscriptionStore from '../../store/subscriptionStore';
import { authEvents } from '../../store/authEvents';
import { stageNativeWardrobeImage, removePendingWardrobeImage } from '../../src/services/pendingWardrobeMedia';
import type { WardrobeState } from '../../store/wardrobeStore';
jest.mock('../../store/auth', () => ({ __esModule:true,default:{getState:()=>({user:{id:'capacity-owner'}})} }));
jest.mock('../../lib/supabase', () => ({ supabase:{} }));
jest.mock('../../src/services/pendingWardrobeMedia', () => ({ stageNativeWardrobeImage:jest.fn(),removePendingWardrobeImage:jest.fn().mockResolvedValue(undefined) }));
const end=Date.parse('2026-10-21T12:00:00Z');
function fixture(count:number) {
 return createStore<WardrobeState>()((set,get,api)=>({...createItemsSlice(set,get,api),
  items:Array.from({length:count},(_,i)=>({id:`existing-${i}`,userId:'capacity-owner'})),
  pendingActions:[],outfits:[],syncToServer:jest.fn().mockResolvedValue(undefined),
 } as unknown as WardrobeState));
}
beforeEach(()=>{
 jest.useFakeTimers();jest.setSystemTime(end-1000);jest.clearAllMocks();authEvents.emitLogin('capacity-owner');
 useSubscriptionStore.setState({tier:'free',effectiveTier:'free',expiryDate:null,trialStartedAt:null,isTrialPending:false});
});
afterEach(()=>jest.useRealTimers());
const remote={category:'top',imageUrl:'https://example.invalid/photo.jpg',userId:'capacity-owner'} as never;
test('an active original trial grants the advertised premium wardrobe capacity',async()=>{
 useSubscriptionStore.setState({trialStartedAt:new Date(end-7*86400000).toISOString()});
 const state=fixture(20);await state.getState().addItem(remote);
 expect(state.getState().items).toHaveLength(21);expect(state.getState().pendingActions).toHaveLength(1);
});
test.each(['premium','vip','lite'] as const)('an expired %s plan cannot bypass the free wardrobe cap',async tier=>{
 useSubscriptionStore.setState({tier,effectiveTier:tier,expiryDate:new Date(end).toISOString()});jest.setSystemTime(end);
 const state=fixture(20);await expect(state.getState().addItem(remote)).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED',limit:20});
 expect(state.getState().items).toHaveLength(20);expect(state.getState().pendingActions).toEqual([]);expect(state.getState().syncToServer).not.toHaveBeenCalled();
});
test('expiry during photo staging rechecks capacity and removes only unpublished files',async()=>{
 useSubscriptionStore.setState({tier:'premium',effectiveTier:'premium',expiryDate:new Date(end).toISOString()});
 let finish!:(uri:string)=>void;
 (stageNativeWardrobeImage as jest.Mock).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;})).mockResolvedValueOnce('file:///owned-preview.jpg');
 const state=fixture(20);
 const pending=state.getState().addItem({...remote as object,imageUrl:'file:///picker.jpg',thumbnailUrl:'file:///preview.jpg'} as never);
 jest.setSystemTime(end);finish('file:///owned-original.jpg');
 await expect(pending).rejects.toMatchObject({code:'WARDROBE_LIMIT_REACHED',limit:20});
 expect(removePendingWardrobeImage).toHaveBeenCalledTimes(2);expect(state.getState().items).toHaveLength(20);expect(state.getState().pendingActions).toEqual([]);
});
