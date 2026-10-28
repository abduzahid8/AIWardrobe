import React from 'react';
import {Text} from 'react-native';
import {act,render,renderHook} from '@testing-library/react-native';
import {useSubscriptionGate} from '../../src/hooks/useSubscriptionGate';
const mockNavigate=jest.fn(),mockConsume=jest.fn(),mockAccess=jest.fn(()=>true);
let mockPending=false,mockRevision=1;
jest.mock('@react-navigation/native',()=>({useNavigation:()=>({navigate:mockNavigate})}));
jest.mock('../../store/authEvents',()=>({authEvents:{getRevision:()=>mockRevision}}));
jest.mock('../../store/subscriptionStore',()=>({__esModule:true,default:(select:any)=>select({tier:'free',effectiveTier:'free',hasActiveSubscription:false,isPremium:false,isTrialPending:mockPending,checkFeatureAccess:mockAccess}),DAILY_QUOTA_FEATURES:['homeSwipes'],TIER_FEATURES:{free:{homeSwipes:3}}}));
jest.mock('../../store/dailyUsageStore',()=>({__esModule:true,default:require('zustand').create((_set:any,get:any)=>({snapshot:{date:'today',counts:{homeSwipes:0}},consume:(...args:any[])=>mockConsume(...args),getRemaining:()=>3-get().snapshot.counts.homeSwipes,getUsed:()=>get().snapshot.counts.homeSwipes,canUse:()=>get().snapshot.counts.homeSwipes<3}))}));
beforeEach(()=>{jest.clearAllMocks();mockPending=false;mockRevision=1;mockAccess.mockReturnValue(true);require('../../store/dailyUsageStore').default.setState({snapshot:{date:'today',counts:{homeSwipes:0}}})});
test('pending trial UI cannot bypass an explicitly verified daily allowance',async()=>{
 mockPending=true;mockConsume.mockResolvedValue({allowed:true,used:1,remaining:2});const{result}=renderHook(()=>useSubscriptionGate());let allowed;
 await act(async()=>{allowed=await result.current.consumeOrPaywall('homeSwipes',{requireServer:true})});expect(allowed).toBe(true);expect(mockConsume).toHaveBeenCalledWith('homeSwipes',1,{requireServer:true});
});
test('a server denial opens the paywall and returns false',async()=>{
 mockConsume.mockResolvedValue({allowed:false,used:3,remaining:0});const{result}=renderHook(()=>useSubscriptionGate());let allowed;
 await act(async()=>{allowed=await result.current.consumeOrPaywall('homeSwipes',{requireServer:true})});expect(allowed).toBe(false);expect(mockNavigate).toHaveBeenCalledWith('Paywall');
});
test('verification errors are not presented as a payment denial',async()=>{
 mockConsume.mockRejectedValue(Error('offline'));const{result}=renderHook(()=>useSubscriptionGate());let failed=false;
 await act(async()=>{try{await result.current.consumeOrPaywall('homeSwipes',{requireServer:true})}catch{failed=true}});expect(failed).toBe(true);expect(mockNavigate).not.toHaveBeenCalled();
});
test('a denied old-account reservation cannot open a paywall in the new account',async()=>{
 let finish!:(value:unknown)=>void;mockConsume.mockImplementation(()=>new Promise(resolve=>finish=resolve));const{result}=renderHook(()=>useSubscriptionGate());const pending=result.current.consumeOrPaywall('homeSwipes',{requireServer:true});mockRevision++;
 let allowed;await act(async()=>{finish({allowed:false,used:3,remaining:0});allowed=await pending});expect(allowed).toBe(false);expect(mockNavigate).not.toHaveBeenCalled();
});

test('a cancelled reservation cannot open a paywall after leaving the card',async()=>{
 const cancel=new AbortController();let finish!:(value:unknown)=>void;mockConsume.mockImplementation(()=>new Promise(resolve=>finish=resolve));const{result}=renderHook(()=>useSubscriptionGate());const pending=result.current.consumeOrPaywall('homeSwipes',{requireServer:true,cancelSignal:cancel.signal});cancel.abort();
 let allowed;await act(async()=>{finish({allowed:false,used:3,remaining:0});allowed=await pending});expect(allowed).toBe(false);expect(mockNavigate).not.toHaveBeenCalled();
});

 test('quota labels rerender when the counter snapshot changes',()=>{
  function Counter(){const gate=useSubscriptionGate();return React.createElement(Text,null,String(gate.getRemaining('homeSwipes')))}
  const view=render(React.createElement(Counter));expect(view.getByText('3')).toBeTruthy();
  act(()=>require('../../store/dailyUsageStore').default.setState({snapshot:{date:'today',counts:{homeSwipes:2}}}));expect(view.getByText('1')).toBeTruthy();
 });
