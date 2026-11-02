jest.mock('../../hooks/useNetworkRecovery', () => ({ useNetworkRecovery: jest.fn() }));
import {act, renderHook, waitFor} from '@testing-library/react-native';
import {useShopCatalog} from '../../hooks/useShopCatalog';
import {supabase} from '../../lib/supabase';
jest.mock('../../lib/supabase', () => ({supabase:{from:jest.fn()}}));
jest.mock('../../data/shopCatalogItems', () => ({ SHOP_CATALOG_ITEMS: [{id: 'seed-fixture', name: 'Shirt', garmentType: 'upper_body'}], SHOP_OUTFITS: [] }));
jest.mock('../../data/inspoMensShopItems', () => ({ CLASSIC_MENS_ITEMS: [] }));
jest.mock('../../src/utils/shopCatalogOrder', () => ({spreadSimilarCatalogItems:(items:unknown[])=>items}));
jest.mock('../../src/utils/imageUrl', () => ({getThumbnailUrl:(url:string)=>url}));
const row=(id:string)=>({id,name:id,price:150,currency:'USD',garment_type:'upper_body',image_url:'https://fixture.invalid/image.jpg'});
function query(result:Promise<any>) {
 const q:any={then:result.then.bind(result)};
 for(const method of ['select','eq','order','range','or','abortSignal']) q[method]=jest.fn(()=>q);
 return q;
}
test('search is sent to the database and paginates matching results',async()=>{
 const first=query(Promise.resolve({data:Array.from({length:100},(_,i)=>row('tuxedo-'+i)),error:null}));
 const second=query(Promise.resolve({data:[row('tuxedo-100')],error:null}));
 (supabase.from as jest.Mock).mockReturnValueOnce(first).mockReturnValueOnce(second);
 const hook=renderHook(()=>useShopCatalog({source:'search-fixture',search:'tuxedo'}));
 await waitFor(()=>expect(hook.result.current.items).toHaveLength(100));
 expect(first.or).toHaveBeenCalledWith('name.ilike."%tuxedo%",brand.ilike."%tuxedo%",description.ilike."%tuxedo%"');
 act(()=>hook.result.current.loadMore());
 await waitFor(()=>expect(hook.result.current.items).toHaveLength(101));
 expect(second.range).toHaveBeenCalledWith(100,199);
 expect(hook.result.current.hasMore).toBe(false);
});
test('an older slow search cannot overwrite a newer search',async()=>{
 let finish!:(value:any)=>void;
 const old=query(new Promise(resolve=>{finish=resolve;}));
 const fresh=query(Promise.resolve({data:[row('latest-result')],error:null}));
 (supabase.from as jest.Mock).mockReturnValueOnce(old).mockReturnValueOnce(fresh);
 const hook=renderHook<ReturnType<typeof useShopCatalog>, {search:string}>(({search})=>useShopCatalog({source:'race-fixture',search}),{initialProps:{search:'old'}});
 hook.rerender({search:'new'});
 await waitFor(()=>expect(hook.result.current.items[0]?.id).toBe('latest-result'));
 await act(async()=>{finish({data:[row('stale-result')],error:null});});
 expect(hook.result.current.items[0].id).toBe('latest-result');
});

test('a failed background catalog refresh keeps the bundled clothing visible', async () => {
 const failed = query(Promise.resolve({data:null, error:{message:'catalog service unavailable'}}));
 (supabase.from as jest.Mock).mockReturnValue(failed);
 const hook = renderHook(() => useShopCatalog({source:'all', category:'tops'}));
 await waitFor(() => expect(hook.result.current.error).toBeTruthy());
 expect(hook.result.current.items.length).toBeGreaterThan(0);
 expect(hook.result.current.loading).toBe(false);
});


test('retries after connectivity returns while a catalog request is still failing', async () => {
 let finish!: (value:any)=>void;
 const old = query(new Promise(resolve=>{finish=resolve;}));
 const fresh = query(Promise.resolve({data:[row('recovered-catalog')],error:null}));
 (supabase.from as jest.Mock).mockReset().mockReturnValueOnce(old).mockReturnValueOnce(fresh);
 const recovery = require('../../hooks/useNetworkRecovery').useNetworkRecovery as jest.Mock;
 const hook = renderHook(() => useShopCatalog({source:'recovery-race-fixture'}));
 await waitFor(()=>expect(supabase.from).toHaveBeenCalledTimes(1));
 act(()=>recovery.mock.calls[recovery.mock.calls.length-1][0]());
 await act(async()=>{finish({data:null,error:{message:'service unavailable'}});});
 await waitFor(()=>expect(hook.result.current.items[0]?.id).toBe('recovered-catalog'));
 expect(supabase.from).toHaveBeenCalledTimes(2);
});
