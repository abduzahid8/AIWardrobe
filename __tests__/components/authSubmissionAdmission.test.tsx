import React from 'react';
import { act, fireEvent, render } from '@testing-library/react-native';
import SignInScreen from '../../screens/SignInScreen';
import SignUpScreen from '../../screens/SignUpScreen';
const mockLogin=jest.fn(),mockRegister=jest.fn(),mockApple=jest.fn(),mockGoogle=jest.fn();
jest.mock('@react-navigation/native',()=>({useNavigation:()=>({navigate:jest.fn()})}));
jest.mock('../../store/auth',()=>({__esModule:true,default:()=>({login:mockLogin,register:mockRegister,signInWithApple:mockApple,signInWithGoogle:mockGoogle})}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string)=>key})}));
jest.mock('../../hooks/useAccessibility',()=>({useAccessibility:()=>({isReducedMotionEnabled:true})}));
jest.mock('react-native-reanimated',()=>({__esModule:true,default:{View:jest.requireActual('react-native').View},FadeInDown:{}}));
jest.mock('expo-apple-authentication',()=>({isAvailableAsync:jest.fn().mockResolvedValue(true),AppleAuthenticationButtonType:{SIGN_IN:'in',SIGN_UP:'up'},AppleAuthenticationButtonStyle:{BLACK:'black'},AppleAuthenticationButton:(props:object)=>require('react').createElement(require('react-native').TouchableOpacity,{...props,testID:'fixtureAppleButton'})}));
jest.mock('expo-linear-gradient',()=>({LinearGradient:jest.requireActual('react-native').View}));
jest.mock('react-native-safe-area-context',()=>({SafeAreaView:jest.requireActual('react-native').View}));
jest.mock('@expo/vector-icons',()=>({Ionicons:jest.requireActual('react-native').View}));
jest.mock('../../components/ui/ScaledText',()=>({ScaledText:jest.requireActual('react-native').Text}));
const cases=[{name:'Sign In',Component:SignInScreen,google:'googleSignInButton',email:'signInButton',action:mockLogin},{name:'Sign Up',Component:SignUpScreen,google:'googleSignUpButton',email:'signUpSubmitButton',action:mockRegister}];
function onPress(view:ReturnType<typeof render>,id:string):()=>unknown{
 let node:ReturnType<typeof view.getByTestId>|null=view.getByTestId(id);
 while(node&&typeof node.props.onPress!=='function')node=node.parent;
 if(!node)throw new Error('Missing press handler');return node.props.onPress;
}
async function form(Component:React.ComponentType){const view=render(<Component/>);await act(async()=>{await Promise.resolve();});return view;}
function hold(){let finish!:()=>void;const promise=new Promise<void>(resolve=>{finish=resolve;});return{promise,finish};}
beforeEach(()=>{jest.useFakeTimers();for(const mock of [mockLogin,mockRegister,mockApple,mockGoogle])mock.mockReset().mockResolvedValue(undefined);});
afterEach(()=>{jest.clearAllTimers();jest.useRealTimers();});
for(const c of cases){
 for(const provider of ['Google','Apple'] as const)test(`${c.name}: same-render duplicate ${provider} presses admit one action`,async()=>{
  const held=hold(),mock=provider==='Google'?mockGoogle:mockApple;mock.mockReturnValue(held.promise);const view=await form(c.Component),press=onPress(view,provider==='Google'?c.google:'fixtureAppleButton');
  try{act(()=>{press();press();});expect(mock).toHaveBeenCalledTimes(1);}finally{await act(async()=>{held.finish();});view.unmount();}
 });
 test(`${c.name}: Google admission blocks an already-captured Apple callback`,async()=>{
  const held=hold();mockGoogle.mockReturnValue(held.promise);mockApple.mockReturnValue(held.promise);const view=await form(c.Component),google=onPress(view,c.google),apple=onPress(view,'fixtureAppleButton');
  try{act(()=>{google();apple();});expect(mockGoogle).toHaveBeenCalledTimes(1);expect(mockApple).not.toHaveBeenCalled();}finally{await act(async()=>{held.finish();});view.unmount();}
 });
 test(`${c.name}: cancellation releases admission for an explicit retry`,async()=>{
  mockGoogle.mockRejectedValueOnce(new Error('cancelled fixture'));const view=await form(c.Component);
  await act(async()=>{onPress(view,c.google)();await Promise.resolve();});await act(async()=>{onPress(view,c.google)();await Promise.resolve();});expect(mockGoogle).toHaveBeenCalledTimes(2);view.unmount();
 });
 test(`${c.name}: repeated email submission in one render admits one action`,async()=>{
  const held=hold();c.action.mockReturnValue(held.promise);const view=await form(c.Component);
  fireEvent.changeText(view.getByTestId(c.name==='Sign In'?'emailInputNative':'signUpEmailInput'),'readiness-ui@example.invalid');
  if(c.name==='Sign In')fireEvent.press(view.getByTestId('continueEmailButton'));
  fireEvent.changeText(view.getByTestId(c.name==='Sign In'?'passwordInputNative':'signUpPasswordInput'),'LocalFixture9');
  if(c.name==='Sign Up')fireEvent.changeText(view.getByTestId('signUpUsernameInput'),'fixture_user');
  const press=onPress(view,c.email);
  try{act(()=>{press();press();});expect(c.action).toHaveBeenCalledTimes(1);}finally{await act(async()=>{held.finish();});view.unmount();}
 });
}
