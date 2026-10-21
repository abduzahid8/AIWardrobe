import React from 'react';
import { Alert } from 'react-native';
import { act, render, fireEvent, waitFor } from '@testing-library/react-native';
import { authEvents } from '../../store/authEvents';
import queue from '../../src/store/uploadQueueStore';
import processor from '../../src/services/uploadQueueProcessor';
import { PendingScanQueue } from '../../components/wardrobe/PendingScanQueue';
jest.mock('../../src/services/uploadQueueProcessor',()=>({__esModule:true,default:{retryUpload:jest.fn(),discardUpload:jest.fn()}}));
jest.mock('react-i18next',()=>({useTranslation:()=>({t:(key:string,options?:any)=>{
  const value=key.split('.').reduce((node:any,part)=>node?.[part],require('../../i18n/locales/en.json'));
  return typeof value==='string'?value.replace('{{index}}',String(options?.index??'')):key;
}})}));
beforeEach(()=>{
  jest.clearAllMocks();authEvents.emitLogin('owner');
  queue.setState({pendingUploads:[{id:'mine',ownerId:'owner',uri:'file:///private.jpg',type:'image',addedAt:1,status:'failed',retryCount:3,error:'private diagnostic'},
    {id:'foreign',ownerId:'other',uri:'file:///other-private.mp4',type:'video',addedAt:1,status:'pending',retryCount:0}]});
  (processor.retryUpload as jest.Mock).mockResolvedValue(true);(processor.discardUpload as jest.Mock).mockResolvedValue(true);
});
afterEach(()=>jest.restoreAllMocks());
it('shows only the authenticated owner and offers a working retry for an exhausted scan',async()=>{
  const view=render(<PendingScanQueue/>);
  expect(view.getByText('Photo 1 · Needs attention')).toBeTruthy();expect(view.queryByText(/Video/)).toBeNull();
  expect(view.queryByText('private diagnostic')).toBeNull();fireEvent.press(view.getByLabelText('Retry: Photo 1'));
  await waitFor(()=>expect(processor.retryUpload).toHaveBeenCalledWith('mine'));
});
it('removal requires the app confirmation before invoking discard',async()=>{
  const alert=jest.spyOn(Alert,'alert');const view=render(<PendingScanQueue/>);
  fireEvent.press(view.getByLabelText('Remove: Photo 1'));expect(processor.discardUpload).not.toHaveBeenCalled();
  await act(async()=>{alert.mock.calls[0][2]![1].onPress!();});
  expect(processor.discardUpload).toHaveBeenCalledWith('mine');
});
it('a confirmation opened by the previous account cannot discard the next account scan',async()=>{
  const alert=jest.spyOn(Alert,'alert');const view=render(<PendingScanQueue/>);
  fireEvent.press(view.getByLabelText('Remove: Photo 1'));
  const confirm=alert.mock.calls[0][2]![1].onPress!;
  act(()=>authEvents.emitLogin('other'));
  await act(async()=>{confirm();});expect(processor.discardUpload).not.toHaveBeenCalled();
  expect(view.getByText('Video 1 · Waiting')).toBeTruthy();
});
it('busy scan state disables queue actions',()=>{
  const view=render(<PendingScanQueue disabled/>);fireEvent.press(view.getByLabelText('Retry: Photo 1'));
  expect(processor.retryUpload).not.toHaveBeenCalled();
});
