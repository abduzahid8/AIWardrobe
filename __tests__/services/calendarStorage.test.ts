import AsyncStorage from '@react-native-async-storage/async-storage';
import { readCalendarLogs, updateCalendarLogs, drainCalendarWrites } from '../../src/services/calendarStorage';
import { authEvents } from '../../store/authEvents';
import { createOutfitLog } from '../../features/calendar/types';

const log = (date: string) => createOutfitLog(date, [{ id: date, type: 'top', image: 'file://top.png' }], 'casual');
beforeEach(async () => {
  await drainCalendarWrites();
  jest.clearAllMocks();
  await AsyncStorage.clear();
  authEvents.emitLogin('calendar-owner');
});

test('simultaneous saves and deletion preserve unrelated calendar dates', async () => {
  await updateCalendarLogs(logs => { logs['2026-10-14'] = log('2026-10-14'); });
  await Promise.all([
    updateCalendarLogs(logs => { logs['2026-10-15'] = log('2026-10-15'); }),
    updateCalendarLogs(logs => { logs['2026-10-16'] = log('2026-10-16'); }),
    updateCalendarLogs(logs => { delete logs['2026-10-14']; }),
  ]);
  expect(Object.keys(await readCalendarLogs()).sort()).toEqual(['2026-10-15', '2026-10-16']);
});

test('a failed write rejects without poisoning subsequent saves', async () => {
  (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
  await expect(updateCalendarLogs(logs => { logs.failed = log('2026-10-15'); })).rejects.toThrow('disk full');
  await updateCalendarLogs(logs => { logs.saved = log('2026-10-16'); });
  expect(Object.keys(await readCalendarLogs())).toEqual(['saved']);
});

test('corrupt calendar is reported and preserved instead of silently discarded', async () => {
  await AsyncStorage.setItem('outfitLogs::calendar-owner', '[]');
  await expect(updateCalendarLogs(logs => { logs.saved = log('2026-10-16'); })).rejects.toThrow('Invalid calendar');
  expect(await AsyncStorage.getItem('outfitLogs::calendar-owner')).toBe('[]');
});

test('queued save cannot cross an account transition', async () => {
  let release!: (value: string | null) => void;
  (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(() => new Promise<string | null>(resolve => { release = resolve; }));
  const pending = updateCalendarLogs(logs => { logs.private = log('2026-10-15'); });
  await Promise.resolve();
  authEvents.emitLogout();
  release(null);
  await expect(pending).rejects.toThrow('Calendar account changed');
  expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});


test('save prepared before logout is rejected even if queued after the account changes', async () => {
  const revision = authEvents.getRevision();
  authEvents.emitLogout();
  await expect(updateCalendarLogs(logs => { logs.private = log('2026-10-15'); }, revision))
    .rejects.toThrow('Calendar account changed');
  expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});

test('direct account switches use independent durable calendar namespaces', async () => {
 await updateCalendarLogs(logs=>{logs['2026-10-15']=log('2026-10-15');});
 authEvents.emitLogin('owner-b');
 expect(await readCalendarLogs()).toEqual({});
 await updateCalendarLogs(logs=>{logs['2026-10-16']=log('2026-10-16');});
 expect(Object.keys(await readCalendarLogs())).toEqual(['2026-10-16']);
 authEvents.emitLogin('calendar-owner');
 expect(Object.keys(await readCalendarLogs())).toEqual(['2026-10-15']);
});

test('unowned legacy plans are neither revealed nor overwritten by a new account save', async () => {
 const legacy=JSON.stringify({'2026-10-14':log('2026-10-14')});
 await AsyncStorage.setItem('outfitLogs',legacy);
 expect(await readCalendarLogs()).toEqual({});
 await updateCalendarLogs(logs=>{logs['2026-10-15']=log('2026-10-15');});
 expect(Object.keys(await readCalendarLogs())).toEqual(['2026-10-15']);
 expect(await AsyncStorage.getItem('outfitLogs')).toBe(legacy);
});

test('anonymous reads disclose no plans and anonymous writes are rejected', async () => {
 await updateCalendarLogs(logs=>{logs.private=log('2026-10-15');});
 authEvents.emitLogout();jest.clearAllMocks();
 expect(await readCalendarLogs()).toEqual({});
 await expect(updateCalendarLogs(logs=>{logs.invalid=log('2026-10-16');})).rejects.toThrow('Calendar account changed');
 expect(AsyncStorage.getItem).not.toHaveBeenCalled();
 expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});

test('an old write completing after account switch cannot affect the new owner key', async () => {
 const write=AsyncStorage.setItem as jest.Mock, original=write.getMockImplementation()!;
 let finish!:()=>void, started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 write.mockImplementationOnce((key:string,value:string)=>new Promise<void>(resolve=>{finish=()=>{Promise.resolve(original(key,value)).then(()=>resolve());};started();}));
 const old=updateCalendarLogs(logs=>{logs.old=log('2026-10-15');});
 const rejection=expect(old).rejects.toThrow('Calendar account changed');
 await active;authEvents.emitLogin('owner-b');
 const fresh=updateCalendarLogs(logs=>{logs.fresh=log('2026-10-16');});
 finish();await Promise.all([rejection,fresh]);
 expect(Object.keys(await readCalendarLogs())).toEqual(['fresh']);
 expect(Object.keys(JSON.parse((await AsyncStorage.getItem('outfitLogs::calendar-owner'))!))).toEqual(['old']);
});

test('a late read is rejected after an account change', async () => {
 let finish!:(value:string)=>void, started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{finish=resolve;started();}));
 const read=readCalendarLogs();const rejection=expect(read).rejects.toThrow('Calendar account changed');
 await active;authEvents.emitLogin('owner-b');
 finish(JSON.stringify({private:log('2026-10-15')}));await rejection;
});

test('a mutation callback cannot write after changing the active identity', async () => {
 await expect(updateCalendarLogs(logs=>{logs.private=log('2026-10-15');authEvents.emitLogout();})).rejects.toThrow('Calendar account changed');
 expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});

test('owner identifiers are encoded into unambiguous storage keys', async () => {
 authEvents.emitLogin('owner::with/slashes');
 await updateCalendarLogs(logs=>{logs.mine=log('2026-10-15');});
 expect(await AsyncStorage.getItem('outfitLogs::owner%3A%3Awith%2Fslashes')).not.toBeNull();
 authEvents.emitLogin('owner%3A%3Awith%2Fslashes');
 expect(await readCalendarLogs()).toEqual({});
});
