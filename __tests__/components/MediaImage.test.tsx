import React from 'react';
import { act, render } from '@testing-library/react-native';
import { Image, ExpoImage } from '../../src/components/MediaImage';
import useAuthStore from '../../store/auth';
import { authEvents } from '../../store/authEvents';
import { Image as NativeImage, Platform } from 'react-native';
jest.mock('../../lib/supabase', () => ({ supabase: {} }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: jest.requireActual('zustand').create(() => ({ session: null })) }));
const owner = '11111111-1111-4111-8111-111111111111';
const uri = `https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/image/original.jpg`;
beforeEach(() => {
  authEvents.emitLogin(owner);
  useAuthStore.setState({ session: { user: { id: owner }, access_token: 'first-token' } as never });
});

it.each([Image, ExpoImage])('private renderer reacts to session refresh and immediate logout', Component => {
  const view = render(<Component testID="photo" source={{ uri }} />);
  const source = () => view.getByTestId('photo').props.source;
  expect(JSON.stringify(source())).toContain('Bearer first-token');
  act(() => { useAuthStore.setState({ session: { user: { id: owner }, access_token: 'second-token' } as never }); });
  expect(JSON.stringify(source())).toContain('Bearer second-token');
  // Auth store may still contain the session while SDK sign-out awaits network.
  act(() => { authEvents.emitLogout(); });
  expect(source() == null || (Array.isArray(source()) && source().length === 0)).toBe(true);
  act(() => {
    const other = '22222222-2222-4222-8222-222222222222';
    authEvents.emitLogin(other);
    useAuthStore.setState({ session: { user: { id: other }, access_token: 'other-token' } as never });
  });
  expect(source() == null || (Array.isArray(source()) && source().length === 0)).toBe(true);
});

test('native dimensions use current private headers and stop after logout', async () => {
  const size = jest.spyOn(NativeImage, 'getSizeWithHeaders').mockImplementation(() => Promise.resolve({ width: 20, height: 30 }));
  try {
    await expect(Image.getSize(uri)).resolves.toEqual({ width: 20, height: 30 });
    expect(size).toHaveBeenCalledWith(uri, { Authorization: 'Bearer first-token' });
    authEvents.emitLogout();
    await expect(Image.getSize(uri)).rejects.toThrow('Private image unavailable');
    expect(size).toHaveBeenCalledTimes(1);
  } finally { size.mockRestore(); }
});

it.each([Image, ExpoImage])('remounts the native view when credentials refresh', Component => {
  const view=render(<Component testID="photo" source={{uri}} />);
  const previous=view.getByTestId('photo');
  act(()=>useAuthStore.setState({session:{user:{id:owner},access_token:'renewed-token'} as never}));
  expect(view.getByTestId('photo')).not.toBe(previous);
});
it.each([Image, ExpoImage])('clears previous native content when the resource changes', Component => {
  const view=render(<Component testID="photo" source={{uri}} />),previous=view.getByTestId('photo');
  view.rerender(<Component testID="photo" source={{uri:uri.replace('original.jpg','different.jpg')}} />);
  expect(view.getByTestId('photo')).not.toBe(previous);
});
it.each([Image, ExpoImage])('keeps the view stable for equivalent source objects', Component => {
  const view=render(<Component testID="photo" source={{uri}} />),previous=view.getByTestId('photo');
  view.rerender(<Component testID="photo" source={{uri}} />);
  expect(view.getByTestId('photo')).toBe(previous);
});

test('Android forwards private headers using the installed SDK array-source path',()=>{
 const replaced=jest.replaceProperty(Platform,'OS','android');
 try{
  const view=render(<Image testID="android-photo" source={{uri}} />);
  expect(view.getByTestId('android-photo').props.source).toEqual([expect.objectContaining({uri,headers:expect.objectContaining({Authorization:'Bearer first-token'})})]);
 }finally{replaced.restore();}
});
