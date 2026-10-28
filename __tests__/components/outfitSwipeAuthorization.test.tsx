import React from 'react';
import {act,render} from '@testing-library/react-native';
import {OutfitSwipeStack,type OutfitSwipeStackHandle,type SwipeStackCard} from '../../components/ui/OutfitSwipeStack';
import {withTiming} from 'react-native-reanimated';
jest.mock('react-native-reanimated',()=>{const React=require('react'),View=require('react-native').View;return{__esModule:true,default:{View},useSharedValue:(value:unknown)=>React.useRef({value}).current,useAnimatedStyle:(fn:()=>unknown)=>fn(),withSpring:jest.fn((value:unknown)=>value),withTiming:jest.fn((value:unknown,_config:unknown,finish?:(done:boolean)=>void)=>{finish?.(true);return value}),runOnJS:(fn:unknown)=>fn,interpolate:()=>0,Extrapolation:{CLAMP:'clamp'}}});
jest.mock('react-native-gesture-handler',()=>({Gesture:{Pan:()=>{const gesture:any={};for(const key of ['enabled','activeOffsetX','failOffsetY','onUpdate','onEnd'])gesture[key]=()=>gesture;return gesture}},GestureDetector:({children}:any)=>children}));
jest.mock('expo-haptics',()=>({impactAsync:jest.fn().mockResolvedValue(undefined),ImpactFeedbackStyle:{Medium:'medium',Light:'light'}}));
jest.mock('@expo/vector-icons',()=>({Ionicons:()=>null}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string)=>key})}));
jest.mock('../../components/ui/ScaledText',()=>({ScaledText:require('react-native').Text}));
jest.mock('../../components/ui/OutfitCollagePreview',()=>({OutfitCollagePreview:()=>null}));
jest.mock('../../src/components/MediaImage',()=>({ExpoImage:{prefetch:jest.fn().mockResolvedValue(undefined)}}));
const cards:SwipeStackCard[]=[{key:'a',caption:'Alpha',items:[]},{key:'b',caption:'Beta',items:[]}];
function deferred(){let resolve!:(value:boolean)=>void;return{promise:new Promise<boolean>(done=>resolve=done),resolve:(value:boolean)=>resolve(value)}}
beforeEach(()=>jest.clearAllMocks());
function Harness({gate,saved,handle}:{gate:()=>Promise<boolean>,saved:()=>void,handle:React.RefObject<OutfitSwipeStackHandle|null>}){
 const[index,setIndex]=React.useState(0);return <OutfitSwipeStack ref={handle} cards={cards} activeIndex={index} beforeSwipe={gate} onSwipeRight={()=>{saved();setIndex(i=>i+1);return true}} onSwipeLeft={()=>{saved();setIndex(i=>i+1);return true}}/>;
}
test('keeps the card visible and blocks duplicate decisions until the allowance is verified',async()=>{
 const pending=deferred(),gate=jest.fn(()=>pending.promise),saved=jest.fn(),handle=React.createRef<OutfitSwipeStackHandle>();const view=render(<Harness gate={gate} saved={saved} handle={handle}/>);
 act(()=>{handle.current!.swipeRight();handle.current!.swipeRight();handle.current!.swipeLeft()});
 expect(gate).toHaveBeenCalledTimes(1);expect(saved).not.toHaveBeenCalled();expect(view.getByTestId('swipe-card-a')).toBeTruthy();expect(view.getByTestId('swipe-authorization-pending')).toBeTruthy();expect(withTiming).not.toHaveBeenCalled();
 await act(async()=>{pending.resolve(true);await Promise.resolve()});
 expect(saved).toHaveBeenCalledTimes(1);expect(view.getByTestId('swipe-card-b')).toBeTruthy();expect(view.queryByTestId('swipe-authorization-pending')).toBeNull();
});
test.each(['denied','unavailable'])('a %s allowance leaves the current card intact and permits a later retry',async mode=>{
 const gate=jest.fn().mockImplementationOnce(()=>mode==='denied'?Promise.resolve(false):Promise.reject(Error('offline'))).mockResolvedValue(true),saved=jest.fn(),handle=React.createRef<OutfitSwipeStackHandle>();const view=render(<Harness gate={gate} saved={saved} handle={handle}/>);
 await act(async()=>{handle.current!.swipeLeft();await Promise.resolve()});expect(saved).not.toHaveBeenCalled();expect(view.getByTestId('swipe-card-a')).toBeTruthy();expect(withTiming).not.toHaveBeenCalled();
 await act(async()=>{handle.current!.swipeRight();await Promise.resolve()});expect(saved).toHaveBeenCalledTimes(1);expect(view.getByTestId('swipe-card-b')).toBeTruthy();
});
test('an approval for an obsolete card cannot act on its replacement',async()=>{
 const pending=deferred(),saved=jest.fn(),handle=React.createRef<OutfitSwipeStackHandle>();const gate=jest.fn((..._args:any[])=>pending.promise);const props={cards,beforeSwipe:gate,onSwipeLeft:saved,onSwipeRight:saved};const view=render(<OutfitSwipeStack ref={handle} activeIndex={0} {...props}/>);
 act(()=>{handle.current!.swipeRight()});view.rerender(<OutfitSwipeStack ref={handle} activeIndex={1} {...props}/>);expect(gate.mock.calls[0][2].aborted).toBe(true);
 await act(async()=>{pending.resolve(true);await Promise.resolve()});expect(saved).not.toHaveBeenCalled();expect(withTiming).not.toHaveBeenCalled();expect(view.getByTestId('swipe-card-b')).toBeTruthy();
});
test('an approval after unmount cannot animate or save',async()=>{
 const pending=deferred(),saved=jest.fn(),handle=React.createRef<OutfitSwipeStackHandle>();const gate=jest.fn((..._args:any[])=>pending.promise);const view=render(<OutfitSwipeStack ref={handle} cards={cards} activeIndex={0} beforeSwipe={gate} onSwipeLeft={saved} onSwipeRight={saved}/>);
 act(()=>{handle.current!.swipeRight()});view.unmount();expect(gate.mock.calls[0][2].aborted).toBe(true);await act(async()=>{pending.resolve(true);await Promise.resolve()});expect(saved).not.toHaveBeenCalled();expect(withTiming).not.toHaveBeenCalled();
});
