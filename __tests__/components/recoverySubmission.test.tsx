import React from 'react';
import { Alert } from 'react-native';
import { act, fireEvent, render } from '@testing-library/react-native';
import ForgotPasswordScreen from '../../screens/ForgotPasswordScreen';
const mockReset=jest.fn();
jest.mock('../../lib/supabase',()=>({supabase:{auth:{resetPasswordForEmail:(...args:unknown[])=>mockReset(...args)}}}));
jest.mock('@react-navigation/native',()=>({useNavigation:()=>({goBack:jest.fn()})}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string)=>key})}));
jest.mock('expo-linking',()=>({createURL:()=> 'aiwardrobe://reset-password'}));
jest.mock('expo-linear-gradient',()=>({LinearGradient:jest.requireActual('react-native').View}));
jest.mock('@expo/vector-icons',()=>({Ionicons:jest.requireActual('react-native').View}));
jest.mock('../../components/ui/ScaledText',()=>({ScaledText:jest.requireActual('react-native').Text}));
function hold(){let finish!:(result:{error:Error|null})=>void;const promise=new Promise<{error:Error|null}>(resolve=>{finish=resolve;});return{promise,finish};}
function form(){const view=render(<ForgotPasswordScreen/>);fireEvent.changeText(view.getByLabelText('auth.email'),'  Readiness@Example.Invalid  ');let node:ReturnType<typeof view.getByLabelText>|null=view.getByLabelText('forgotPasswordScreen.sendResetLink');while(node&&typeof node.props.onPress!=='function')node=node.parent;if(!node)throw new Error('Missing submit callback');return{view,press:node.props.onPress as ()=>unknown};}
beforeEach(()=>{mockReset.mockReset().mockResolvedValue({error:null});jest.spyOn(Alert,'alert').mockImplementation(()=>undefined);});
afterEach(()=>jest.restoreAllMocks());
test('same-render repeated presses send one pending recovery request',async()=>{
 const held=hold();mockReset.mockReturnValue(held.promise);const{view,press}=form();
 try{act(()=>{press();press();});expect(mockReset).toHaveBeenCalledTimes(1);expect(view.getByLabelText('auth.email').props.editable).toBe(false);}finally{await act(async()=>{held.finish({error:null});});view.unmount();}
});
test('confirmation identifies the normalized submitted email despite a queued edit',async()=>{
 const held=hold();mockReset.mockReturnValue(held.promise);const{view,press}=form();const edit=view.getByLabelText('auth.email').props.onChangeText;
 act(()=>{press();edit('other@example.invalid');});await act(async()=>{held.finish({error:null});});
 expect(mockReset).toHaveBeenCalledWith('readiness@example.invalid',{redirectTo:'aiwardrobe://reset-password'});
 expect(view.getByText('readiness@example.invalid')).toBeTruthy();expect(view.queryByText('other@example.invalid')).toBeNull();view.unmount();
});
test('an error arriving after leaving the screen does not show a stale alert',async()=>{
 const held=hold();mockReset.mockReturnValue(held.promise);const{view,press}=form();act(()=>{press();});view.unmount();
 await act(async()=>{held.finish({error:new Error('synthetic recovery failure')});});expect(Alert.alert).not.toHaveBeenCalled();
});
test('a failed request releases admission for an explicit retry',async()=>{
 mockReset.mockResolvedValueOnce({error:new Error('synthetic recovery failure')});const{view,press}=form();
 await act(async()=>{press();await Promise.resolve();});expect(Alert.alert).toHaveBeenCalledTimes(1);
 await act(async()=>{press();await Promise.resolve();});expect(mockReset).toHaveBeenCalledTimes(2);expect(view.getByText('forgotPasswordScreen.checkYourEmail')).toBeTruthy();view.unmount();
});
test('a captured submit callback cannot resend after success',async()=>{
 const{view,press}=form();await act(async()=>{press();await Promise.resolve();});await act(async()=>{press();await Promise.resolve();});expect(mockReset).toHaveBeenCalledTimes(1);view.unmount();
});
