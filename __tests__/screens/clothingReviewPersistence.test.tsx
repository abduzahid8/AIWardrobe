const mockReviewAck=jest.fn(),mockReviewFlush=jest.fn();
let mockReview=false;
jest.mock('../../src/services/scanReviews',()=>({acknowledgeScanReviewItem:(...args:any[])=>mockReviewAck(...args)}));
import React from 'react';
import { Alert, TouchableOpacity } from 'react-native';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import * as Haptics from 'expo-haptics';
import { authEvents } from '../../store/authEvents';
import ClothingDetailEditor from '../../components/ClothingDetailEditor';
import { saveClothingEdits } from '../../src/services/clothingEditorService';
const mockUpdate = jest.fn();
const mockAdd = jest.fn();
let mockEditing = true;
let mockDirect = false;
const mockClassify = jest.fn();
const mockReadImage = jest.fn();
const mockItem = { id: 'owned', category: 'top', subCategory: 't-shirt', primaryColor: 'Black', seasons: ['summer'], occasions: [], imageUrl: '' };
jest.mock('../../src/services/clothingEditorService', () => ({ saveClothingEdits: jest.fn() }));
jest.mock('../../src/services/externalAIService', () => ({ ExternalAIService: { classifyOnly: (...args: unknown[]) => mockClassify(...args) } }));
jest.mock('expo-file-system/legacy', () => ({ readAsStringAsync: (...args: unknown[]) => mockReadImage(...args) }));
jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, flushWardrobePersistence:()=>mockReviewFlush(), default: { getState: () => ({ updateItem: mockUpdate, addItem: mockAdd }) } }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: Object.assign((selector: any) => selector({ user: { id: 'owner' }, session: { user: { id: 'owner' } } }), { getState: () => ({ user: { id: 'owner' }, session: { user: { id: 'owner' } } }) }) }));
jest.mock('@react-navigation/native', () => ({ useNavigation: () => ({ goBack: jest.fn() }), useRoute: () => ({ params: mockDirect ? { imageUri: 'file:///direct.jpg' } : { existingItem: mockEditing ? mockItem : undefined, detectedType: 'tops', ...(mockReview?{scanReviewId:'11111111-1111-4111-8111-111111111111',scanReviewItemId:'44444444-4444-4444-8444-444444444444'}:{}) } }) }));
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('../../components/ui/ScaledText', () => ({ ScaledText: require('react-native').Text }));
beforeEach(() => { jest.clearAllMocks(); mockReview=false;mockReviewAck.mockReset().mockResolvedValue(undefined);mockReviewFlush.mockReset().mockResolvedValue(undefined); mockEditing = true; mockDirect = false; mockClassify.mockReset(); mockReadImage.mockReset(); mockReadImage.mockResolvedValue('photo-base64'); mockAdd.mockReset(); authEvents.emitLogin('owner'); (saveClothingEdits as jest.Mock).mockReset(); });

test('review editor keeps the original item when an existing-item save fails', async () => {
  (saveClothingEdits as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  const view = render(<ClothingDetailEditor />);
  fireEvent.press(view.getByText('clothingEditor.updateItem'));
  await waitFor(() => expect(alert).toHaveBeenCalledWith('clothingEditor.error', 'clothingEditor.failedToSaveItem'));
  expect(mockUpdate).not.toHaveBeenCalled();
  alert.mockRestore(); error.mockRestore();
});

test('review editor confirms server persistence before updating local state', async () => {
  (saveClothingEdits as jest.Mock).mockResolvedValueOnce({ id: 'owned' });
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const view = render(<ClothingDetailEditor />);
  fireEvent.press(view.getByText('clothingEditor.updateItem'));
  await waitFor(() => expect(mockUpdate).toHaveBeenCalledWith('owned', expect.objectContaining({ primaryColor: 'black', seasons: ['summer'] })));
  expect((saveClothingEdits as jest.Mock).mock.calls[0][1]).not.toHaveProperty('subCategory');
  alert.mockRestore();
});

test('same-tick new-item taps share one in-flight save and defer success feedback', async () => {
  mockEditing = false;
  let finish!: () => void;
  mockAdd.mockReturnValueOnce(new Promise<void>(resolve => { finish = resolve; }));
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const view = render(<ClothingDetailEditor />);
  const press = view.UNSAFE_getAllByType(TouchableOpacity).find(button => button.props.testID === 'clothingEditorSave')!.props.onPress;
  let first!: Promise<void>, second!: Promise<void>;
  act(() => { first = press(); second = press(); });
  expect(mockAdd).toHaveBeenCalledTimes(1);
  expect(Haptics.notificationAsync).not.toHaveBeenCalled();
  expect(alert).not.toHaveBeenCalled();
  expect(view.getByTestId('clothingEditorSave').props.accessibilityState).toMatchObject({disabled:true,busy:true});
  await act(async () => { finish(); await Promise.all([first,second]); });
  expect(Haptics.notificationAsync).toHaveBeenCalledTimes(1);
  expect(alert).toHaveBeenCalledWith('clothingEditor.saved','clothingEditor.itemAddedToWardrobe',expect.any(Array));
  alert.mockRestore();
});

test('failed asynchronous custom save reports an error and allows retry without false success', async () => {
  const save = jest.fn().mockRejectedValueOnce(new Error('disk full')).mockResolvedValueOnce(undefined);
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    const view = render(<ClothingDetailEditor onSave={save} />);
    fireEvent.press(view.getByTestId('clothingEditorSave'));
    await waitFor(() => expect(alert).toHaveBeenCalledWith('clothingEditor.error','clothingEditor.failedToSaveItem'));
    expect(Haptics.notificationAsync).not.toHaveBeenCalled();
    await waitFor(() => expect(view.getByTestId('clothingEditorSave').props.accessibilityState.busy).toBe(false));
    fireEvent.press(view.getByTestId('clothingEditorSave'));
    await waitFor(() => expect(Haptics.notificationAsync).toHaveBeenCalledTimes(1));
    expect(save).toHaveBeenCalledTimes(2);expect(saveClothingEdits).not.toHaveBeenCalled();
  } finally {alert.mockRestore();error.mockRestore();}
});

test('late new-item confirmation after logout cannot show success for the next account', async () => {
  mockEditing = false;
  let finish!: () => void;
  mockAdd.mockReturnValueOnce(new Promise<void>(resolve => { finish = resolve; }));
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  const view = render(<ClothingDetailEditor />);
  fireEvent.press(view.getByTestId('clothingEditorSave'));
  await act(async () => {authEvents.emitLogout();authEvents.emitLogin('other');finish();});
  expect(alert).not.toHaveBeenCalled();expect(Haptics.notificationAsync).not.toHaveBeenCalled();
  alert.mockRestore();
});


test('direct-photo editor saves retained analysis metadata after rendering the result', async () => {
  mockEditing = false; mockDirect = true;
  mockClassify.mockResolvedValueOnce({ success: true, description: 'Cotton beige jacket', classification: {
    category: 'jacket', section: 'outerwear', confidence: 0.92, attributes: { color: 'beige', material: 'cotton', style: 'casual' },
  } });
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  try {
    const view = render(<ClothingDetailEditor />);
    await waitFor(() => expect(view.getByTestId('clothingEditorSave').props.accessibilityState.busy).toBe(false));
    fireEvent.press(view.getByTestId('clothingEditorSave'));
    await waitFor(() => expect(mockAdd).toHaveBeenCalledWith(expect.objectContaining({
      userId: 'owner', category: 'outerwear', subCategory: 'jacket', primaryColor: 'beige', material: 'cotton',
      name: 'Cotton beige jacket', detectionConfidence: 0.92, pattern: '',
    })));
    expect(mockReadImage).toHaveBeenCalledWith('file:///direct.jpg', { encoding: 'base64' });
  } finally { alert.mockRestore(); }
});

test('direct-photo analysis failure is visible instead of silently becoming a successful classification', async () => {
  mockEditing = false; mockDirect = true;
  mockClassify.mockResolvedValueOnce({ success: false, classification: null });
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  try {
    render(<ClothingDetailEditor />);
    await waitFor(() => expect(alert).toHaveBeenCalledWith('videoAnalysis.analysisFailed', 'videoAnalysis.somethingWrong'));
    expect(mockAdd).not.toHaveBeenCalled();
  } finally { alert.mockRestore(); }
});


test('save remains blocked until direct-photo classification finishes', async () => {
  mockEditing = false; mockDirect = true;
  let finish!: (value: unknown) => void;
  mockClassify.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  try {
    const view = render(<ClothingDetailEditor />);
    await waitFor(() => expect(mockClassify).toHaveBeenCalledTimes(1));
    expect(view.getByTestId('clothingEditorSave').props.accessibilityState).toMatchObject({ disabled: true, busy: true });
    const press = view.UNSAFE_getAllByType(TouchableOpacity).find(button => button.props.testID === 'clothingEditorSave')!.props.onPress;
    await act(async () => { await press(); });
    expect(mockAdd).not.toHaveBeenCalled();
    await act(async () => { finish({ success: false, classification: null }); });
    await waitFor(() => expect(view.getByTestId('clothingEditorSave').props.accessibilityState.busy).toBe(false));
  } finally { alert.mockRestore(); }
});

test('scan review acknowledgement follows the stable item save and durable wardrobe flush',async()=>{
 mockEditing=false;mockReview=true;mockAdd.mockResolvedValue(undefined);const alert=jest.spyOn(Alert,'alert').mockImplementation(()=>{});
 const view=render(<ClothingDetailEditor/>);fireEvent.press(view.getByTestId('clothingEditorSave'));
 await waitFor(()=>expect(mockReviewAck).toHaveBeenCalledWith('owner','11111111-1111-4111-8111-111111111111','44444444-4444-4444-8444-444444444444'));
 expect(mockAdd).toHaveBeenCalledWith(expect.any(Object),{idempotencyId:'44444444-4444-4444-8444-444444444444'});
 expect(mockReviewFlush.mock.invocationCallOrder[0]).toBeGreaterThan(mockAdd.mock.invocationCallOrder[0]);
 expect(mockReviewAck.mock.invocationCallOrder[0]).toBeGreaterThan(mockReviewFlush.mock.invocationCallOrder[0]);alert.mockRestore();
});
test('failed wardrobe persistence cannot acknowledge the pending scan review',async()=>{
 mockEditing=false;mockReview=true;mockAdd.mockResolvedValue(undefined);mockReviewFlush.mockRejectedValueOnce(new Error('disk full'));
 const alert=jest.spyOn(Alert,'alert').mockImplementation(()=>{});const error=jest.spyOn(console,'error').mockImplementation(()=>{});
 const view=render(<ClothingDetailEditor/>);fireEvent.press(view.getByTestId('clothingEditorSave'));
 await waitFor(()=>expect(alert).toHaveBeenCalledWith('clothingEditor.error','clothingEditor.failedToSaveItem'));
 expect(mockReviewAck).not.toHaveBeenCalled();alert.mockRestore();error.mockRestore();
});
