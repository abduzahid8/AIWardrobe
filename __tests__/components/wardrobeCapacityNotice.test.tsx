import React from 'react';
import { act, fireEvent, render } from '@testing-library/react-native';
import { WardrobeCapacityNotice } from '../../components/WardrobeCapacityNotice';
import useWardrobeStore from '../../store/wardrobeStore';
const mockNavigate=jest.fn(),mockSync=jest.fn();
jest.mock('../../hooks/useAppNavigation',()=>({useAppNavigation:()=>({navigate:mockNavigate})}));
jest.mock('../../store/wardrobeStore',()=>({__esModule:true,default:require('zustand').create(()=>({wardrobeCapacityIssue:null,isSyncing:false,syncToServer:()=>mockSync()}))}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string,values?:{limit?:number})=>key==='wardrobeSync.capacityMessage'?`Kept on this device; limit ${values?.limit}`:key})}));
beforeEach(()=>{jest.clearAllMocks();mockSync.mockResolvedValue(undefined);useWardrobeStore.setState({wardrobeCapacityIssue:{limit:20,actionIds:['blocked']},isSyncing:false});});
test('shows the retained-work guidance and retries the existing queue',()=>{
 const screen=render(<WardrobeCapacityNotice/>);
 expect(screen.getByText('Kept on this device; limit 20')).toBeTruthy();
 fireEvent.press(screen.getByText('wardrobeSync.retry'));expect(mockSync).toHaveBeenCalledTimes(1);
});
test('cannot start overlapping retries while sync runs',()=>{
 useWardrobeStore.setState({isSyncing:true});const screen=render(<WardrobeCapacityNotice/>);
 fireEvent.press(screen.getByText('wardrobeSync.retrying'));expect(mockSync).not.toHaveBeenCalled();
});
test('opens the paywall without dropping queued work',()=>{
 const screen=render(<WardrobeCapacityNotice/>);fireEvent.press(screen.getByText('wardrobeSync.upgrade'));
 expect(mockNavigate).toHaveBeenCalledWith('Paywall');expect(useWardrobeStore.getState().wardrobeCapacityIssue).not.toBeNull();
});
test('notice disappears when confirmed sync clears its issue',()=>{
 const screen=render(<WardrobeCapacityNotice/>);
 act(()=>useWardrobeStore.setState({wardrobeCapacityIssue:null}));expect(screen.queryByText('wardrobeSync.capacityTitle')).toBeNull();
});
test.each(['en','ru','uz'])('%s contains every capacity-notice translation',lang=>{
 const strings=require(`../../i18n/locales/${lang}.json`).wardrobeSync;
 for(const key of ['capacityTitle','capacityMessage','retry','upgrade','retrying'])expect(typeof strings[key]).toBe('string');
 expect(strings.capacityMessage).toContain('{{limit}}');
});
