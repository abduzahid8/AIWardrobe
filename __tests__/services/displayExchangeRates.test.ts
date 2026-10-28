import AsyncStorage from '@react-native-async-storage/async-storage';
import {getDisplayExchangeRates} from '../../src/services/displayExchangeRates';
jest.mock('../../src/services/budgetRates',()=>({getBudgetRates:jest.fn()}));
test('display conversion coalesces a real-format official feed and caches the validated snapshot',async()=>{
 const date=new Date(Date.now()+5*3600000).toISOString().slice(0,10).split('-').reverse().join('.');
 const original=global.fetch;
 const request=jest.fn(async(..._args:Parameters<typeof fetch>)=>new Response(JSON.stringify(['USD','EUR','GBP'].map((Ccy,i)=>({Ccy,Nominal:'1',Rate:String(12000+i*1000),Date:date})))));
 global.fetch=request as typeof fetch;
 try {
  await AsyncStorage.removeItem('public-cbu-display-rates-v1');
  const [a,b]=await Promise.all([getDisplayExchangeRates(),getDisplayExchangeRates()]);
  expect(a).toEqual(b);expect(a.source).toBe('CBU');expect(a.uzsPerUnit.USD).toBe(12000);expect(request).toHaveBeenCalledTimes(1);
  expect(String(request.mock.calls[0][0])).toContain('https://cbu.uz/');
  expect(await getDisplayExchangeRates()).toEqual(a);expect(request).toHaveBeenCalledTimes(1);
 } finally {global.fetch=original;}
});
