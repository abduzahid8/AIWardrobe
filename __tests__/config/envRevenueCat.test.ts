let mockPlatform='ios';
jest.mock('react-native',()=>{
 const native=jest.requireActual('react-native');
 Object.defineProperty(native.Platform,'OS',{configurable:true,get:()=>mockPlatform});
 return native;
});
const keys=['EXPO_PUBLIC_REVENUECAT_API_KEY','EXPO_PUBLIC_REVENUECAT_IOS_API_KEY','EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY'];
const before=Object.fromEntries(keys.map(key=>[key,process.env[key]]));
beforeEach(()=>{for(const key of keys)delete process.env[key];});
afterAll(()=>{for(const key of keys){if(before[key]===undefined)delete process.env[key];else process.env[key]=before[key];}});
function configured(platform:string){
 mockPlatform=platform;let key:string|undefined;
 jest.isolateModules(()=>{key=jest.requireActual('../../src/config/env').Config.revenueCat.apiKey;});
 return key;
}
test('actual Config keeps legacy iOS builds working and does not reuse that key on Android',()=>{
 process.env.EXPO_PUBLIC_REVENUECAT_API_KEY='appl_fixture';
 expect(configured('ios')).toBe('appl_fixture');expect(configured('android')).toBe('');
});
test('actual Config selects the Android key while preserving the iOS key',()=>{
 process.env.EXPO_PUBLIC_REVENUECAT_API_KEY='appl_legacy';
 process.env.EXPO_PUBLIC_REVENUECAT_IOS_API_KEY='appl_specific';
 process.env.EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY='goog_specific';
 expect(configured('ios')).toBe('appl_specific');expect(configured('android')).toBe('goog_specific');
});
