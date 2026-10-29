import { act, renderHook, waitFor } from '@testing-library/react-native';
import { useFocusEffect } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { fetchItemsFromServer } from '../../store/wardrobeSyncService';
import { useOutfitCalendar } from '../../features/calendar/hooks/useOutfitCalendar';
import { authEvents } from '../../store/authEvents';
import type { ClothingItem } from '../../src/types/domain';

jest.mock('@react-navigation/native', () => ({ useFocusEffect: jest.fn() }));
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('../../store/wardrobeSyncService', () => ({ fetchItemsFromServer: jest.fn() }));
jest.mock('expo-asset', () => ({ Asset: { fromModule: jest.fn() } }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: { getState: () => ({ session: { user: { id: 'calendar-owner-a' } } }) } }));
jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, default: { getState: () => ({ items: [
  { id: 'owned-offline', userId: 'calendar-owner-a', category: 'top', imageUrl: 'file://owned.png' },
  { id: 'foreign-offline', userId: 'calendar-owner-b', category: 'top', imageUrl: 'file://foreign.png' },
  { id: 'unowned-legacy', category: 'top', imageUrl: 'file://legacy.png' },
] }) } }));
const fetchItems = fetchItemsFromServer as jest.MockedFunction<typeof fetchItemsFromServer>;
const garment = (id: string) => ({ id, category: 'top', subCategory: 'shirt', primaryColor: 'blue', imageUrl: `https://catalog.test/${id}.png`, name: id } as ClothingItem);
const originalCalendarRead = (AsyncStorage.getItem as jest.Mock).getMockImplementation()!;
let focus: () => void;
beforeEach(async () => {
  jest.clearAllMocks(); await AsyncStorage.clear(); authEvents.emitLogin('calendar-owner-a');
  (useFocusEffect as jest.Mock).mockImplementation(callback => { focus = callback; });
  fetchItems.mockResolvedValue([]);
});

test('calendar retains all garments from the shared paginated reader, including beyond 1000', async () => {
  fetchItems.mockResolvedValue(Array.from({ length: 1251 }, (_, index) => garment(`item-${index}`)));
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => { focus(); });
  await waitFor(() => expect(result.current.wardrobeItems).toHaveLength(1251));
  expect(result.current.wardrobeItems[1250]).toMatchObject({ id: 'item-1250', type: 'shirt', color: 'blue', image: 'https://catalog.test/item-1250.png' });
});

test('an empty confirmed wardrobe clears old choices instead of reviving the legacy cache', async () => {
  fetchItems.mockResolvedValueOnce([garment('old')]).mockResolvedValueOnce([]);
  await AsyncStorage.setItem('myWardrobeItems', JSON.stringify([{ id: 'foreign-legacy' }]));
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => { focus(); });
  await waitFor(() => expect(result.current.wardrobeItems).toHaveLength(1));
  act(() => { focus(); });
  await waitFor(() => expect(result.current.wardrobeItems).toEqual([]));
  expect((AsyncStorage.getItem as jest.Mock).mock.calls.some(([key]) => key === 'myWardrobeItems')).toBe(false);
});

test('offline selection reads only explicitly owned cached garments', async () => {
  fetchItems.mockResolvedValue(null);
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => { focus(); });
  await waitFor(() => expect(result.current.wardrobeItems.map(item => item.id)).toEqual(['owned-offline']));
});

test('an old account response cannot restore wardrobe or selected draft after a direct account switch', async () => {
  let finish!: (items: ClothingItem[]) => void;
  fetchItems.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => {
    result.current.openLogModalForDate('2026-10-17');
    result.current.toggleItemSelection({ id: 'old-selection', type: 'top', image: 'file://old.png' });
    focus();
  });
  act(() => { authEvents.emitLogin('calendar-owner-b'); });
  expect(result.current.selectedItems).toEqual([]);
  expect(result.current.showLogModal).toBe(false);
  expect(result.current.selectedDate).toBeNull();
  await act(async () => { finish([garment('old-private')]); });
  expect(result.current.wardrobeItems).toEqual([]);
});

test('same-account login notifications retain the current draft; logout clears it', () => {
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => {
    result.current.openLogModalForDate('2026-10-17');
    result.current.toggleItemSelection({ id: 'selection', type: 'top', image: 'file://photo.png' });
  });
  act(() => { authEvents.emitLogin('calendar-owner-a'); });
  expect(result.current.selectedItems).toHaveLength(1);
  expect(result.current.showLogModal).toBe(true);
  act(() => { authEvents.emitLogout(); });
  expect(result.current.selectedItems).toEqual([]);
  expect(result.current.showLogModal).toBe(false);
});

test('read failure keeps loaded plans, exposes retry state, and successful retry can show a genuinely empty calendar', async()=>{
 const key='outfitLogs::calendar-owner-a';const logs={'2099-01-01':{date:'2099-01-01',items:[{id:'coat',type:'jacket',image:'file://coat.jpg'}],occasion:'casual'}};
 await AsyncStorage.setItem(key,JSON.stringify(logs));const {result}=renderHook(()=>useOutfitCalendar());
 await act(async()=>{await result.current.reloadOutfitLogs();});expect(result.current.outfitLogs).toEqual(logs);
 const read=jest.spyOn(AsyncStorage,'getItem').mockRejectedValueOnce(Error('temporary read failure'));
 try{await act(async()=>{await result.current.reloadOutfitLogs();});expect(result.current.logsLoadFailed).toBe(true);expect(result.current.outfitLogs).toEqual(logs);}
 finally{read.mockImplementation(originalCalendarRead);}
 await AsyncStorage.setItem(key,'{}');await act(async()=>{await result.current.reloadOutfitLogs();});
 expect(result.current.logsLoadFailed).toBe(false);expect(result.current.outfitLogs).toEqual({});
});

test('an older read failure cannot mark a newer successful refresh as failed',async()=>{
 let reject!:(error:Error)=>void;const held=new Promise<string>((_resolve,r)=>{reject=r;});
 const logs={'2099-01-01':{date:'2099-01-01',items:[{id:'coat'}],occasion:'casual'}};
 const read=jest.spyOn(AsyncStorage,'getItem').mockImplementationOnce(()=>held).mockResolvedValueOnce(JSON.stringify(logs));
 const {result}=renderHook(()=>useOutfitCalendar());let old!:Promise<void>;
 try{
  act(()=>{old=result.current.reloadOutfitLogs();});await act(async()=>{await Promise.resolve();});
  await act(async()=>{await result.current.reloadOutfitLogs();});await act(async()=>{reject(Error('old failure'));await old;});
  expect(result.current.logsLoadFailed).toBe(false);expect(result.current.outfitLogs).toEqual(logs);
 }finally{read.mockImplementation(originalCalendarRead);}
});

test('account change clears prior failure and a late old-account rejection stays ignored',async()=>{
 let reject!:(error:Error)=>void;const held=new Promise<string>((_resolve,r)=>{reject=r;});
 const read=jest.spyOn(AsyncStorage,'getItem').mockImplementationOnce(()=>held);const {result}=renderHook(()=>useOutfitCalendar());let old!:Promise<void>;
 try{
  act(()=>{old=result.current.reloadOutfitLogs();});await act(async()=>{await Promise.resolve();});
  act(()=>{authEvents.emitLogin('calendar-owner-b');});await act(async()=>{reject(Error('old failure'));await old;});
  expect(result.current.logsLoadFailed).toBe(false);expect(result.current.outfitLogs).toEqual({});
 }finally{read.mockImplementation(originalCalendarRead);}
});

test('a read started before a successful save cannot erase the newly saved day',async()=>{
 let finish!:(value:string)=>void;const held=new Promise<string>(resolve=>{finish=resolve;});
 const read=jest.spyOn(AsyncStorage,'getItem').mockImplementationOnce(()=>held);
 const {result}=renderHook(()=>useOutfitCalendar());let old!:Promise<void>;
 try{
  act(()=>{old=result.current.reloadOutfitLogs();});await act(async()=>{await Promise.resolve();});
  act(()=>{result.current.openLogModalForDate('2099-01-01');result.current.toggleItemSelection({id:'coat',type:'jacket',image:'file://coat.jpg'});});
  await act(async()=>{await result.current.saveOutfit();});expect(result.current.outfitLogs['2099-01-01']).toBeDefined();
  await act(async()=>{finish('{}');await old;});expect(result.current.outfitLogs['2099-01-01']).toBeDefined();
 }finally{read.mockImplementation(originalCalendarRead);}
});

test('cached owned choices appear while the server request is still pending', async () => {
  let finish!: (items: ClothingItem[]) => void;
  fetchItems.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const { result } = renderHook(() => useOutfitCalendar());
  act(() => { focus(); });
  expect(result.current.wardrobeItems.map(item => item.id)).toEqual(['owned-offline']);
  await act(async () => { finish([]); });
  expect(result.current.wardrobeItems).toEqual([]);
});
