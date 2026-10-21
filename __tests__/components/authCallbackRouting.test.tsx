import React from 'react';
import {act,render} from '@testing-library/react-native';
import AuthCallbackScreen from '../../screens/AuthCallbackScreen';
const mockReset=jest.fn();
let mockRoutes=['Main','Paywall','AuthCallback'];
let mockAuth={isAuthenticated:false,loading:true,error:null as string|null};
jest.mock('../../store/auth',()=>({__esModule:true,default:(select:(state:typeof mockAuth)=>unknown)=>select(mockAuth)}));
jest.mock('../../hooks/useAppNavigation',()=>({useAppNavigation:()=>({getState:()=>({routeNames:mockRoutes}),reset:mockReset})}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string)=>key})}));
jest.mock('expo-linear-gradient',()=>({LinearGradient:jest.requireActual('react-native').View}));
jest.mock('../../components/ui/ScaledText',()=>({ScaledText:jest.requireActual('react-native').Text}));
beforeEach(()=>{jest.useFakeTimers();mockReset.mockClear();mockRoutes=['Main','Paywall','AuthCallback'];mockAuth={isAuthenticated:false,loading:true,error:null};});
afterEach(()=>{jest.useRealTimers();});
test.each(['Main','StyleQuiz','Paywall'])('an already authenticated callback exits to the registered %s gate',destination=>{
 mockRoutes=[destination,'ResetPassword','AuthCallback'];mockAuth.isAuthenticated=true;mockAuth.loading=false;
 const view=render(<AuthCallbackScreen/>);expect(mockReset).toHaveBeenCalledWith({index:0,routes:[{name:destination}]});view.unmount();
});
test('a callback mounted before login exits after session publication',()=>{
 const view=render(<AuthCallbackScreen/>);expect(mockReset).not.toHaveBeenCalled();
 mockAuth={isAuthenticated:true,loading:false,error:null};view.rerender(<AuthCallbackScreen/>);
 expect(mockReset).toHaveBeenCalledWith({index:0,routes:[{name:'Main'}]});view.unmount();
});
test('a settled callback without a session or error returns to sign-in instead of spinning forever',()=>{
 mockAuth.loading=false;const view=render(<AuthCallbackScreen/>);
 act(()=>jest.advanceTimersByTime(1500));expect(mockReset).toHaveBeenCalledWith({index:0,routes:[{name:'SignIn'}]});view.unmount();
});
test('successful session publication cancels the pending failure redirect',()=>{
 mockAuth.loading=false;const view=render(<AuthCallbackScreen/>);
 mockAuth.isAuthenticated=true;view.rerender(<AuthCallbackScreen/>);act(()=>jest.advanceTimersByTime(1500));
 expect(mockReset).toHaveBeenCalledTimes(1);expect(mockReset).toHaveBeenCalledWith({index:0,routes:[{name:'Main'}]});view.unmount();
});
