import React from 'react';
import { Alert } from 'react-native';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import ClothingDetailScreen from '../../screens/ClothingDetailScreen';
import { saveClothingEdits } from '../../src/services/clothingEditorService';
import { authEvents } from '../../store/authEvents';

const mockUpdateItem = jest.fn();
const mockItem = { id: 'owned-item', userId: 'owner', name: 'Shirt', category: 'top', subCategory: 't-shirt',
  primaryColor: 'Black', colorHex: '#1C1C1E', seasons: [], occasions: [], wearCount: 0, imageUrl: '' };
jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, default: (selector: (state: unknown) => unknown) => selector({
  items: [mockItem], updateItem: mockUpdateItem, removeItem: jest.fn(), toggleFavorite: jest.fn(),
}) }));
jest.mock('expo-haptics', () => ({ selectionAsync: jest.fn(), impactAsync: jest.fn(), notificationAsync: jest.fn(), ImpactFeedbackStyle: { Light: 'light' }, NotificationFeedbackType: { Success: 'success' } }));
jest.mock('../../src/services/clothingEditorService', () => ({ saveClothingEdits: jest.fn() }));
jest.mock('@react-navigation/native', () => ({ useNavigation: () => ({ goBack: jest.fn() }), useRoute: () => ({ params: { itemId: 'owned-item' } }) }));
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('../../components/ui/ScaledText', () => ({ ScaledText: require('react-native').Text }));

beforeEach(() => { jest.clearAllMocks(); (saveClothingEdits as jest.Mock).mockReset(); authEvents.emitLogin('owner'); });

test('failed server save keeps the editor open and leaves the original local item untouched', async () => {
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  (saveClothingEdits as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  const view = render(<ClothingDetailScreen />);
  fireEvent.press(view.getByLabelText('clothingDetail.editItem'));
  fireEvent.press(view.getByText('clothingEditor.types.bottoms'));
  fireEvent.press(view.getByText('clothingEditor.save'));
  await waitFor(() => expect(alert).toHaveBeenCalled());
  expect(mockUpdateItem).not.toHaveBeenCalled();
  expect(view.getByText('clothingEditor.save')).toBeTruthy();
  alert.mockRestore();
});

test('changing type persists both the canonical category and subtype before updating local state', async () => {
  (saveClothingEdits as jest.Mock).mockResolvedValueOnce({ id: 'owned-item' });
  const view = render(<ClothingDetailScreen />);
  fireEvent.press(view.getByLabelText('clothingDetail.editItem'));
  fireEvent.press(view.getByText('clothingEditor.types.bottoms'));
  fireEvent.press(view.getByText('clothingEditor.save'));
  await waitFor(() => expect(mockUpdateItem).toHaveBeenCalledWith('owned-item', expect.objectContaining({ category: 'bottom', subCategory: 'bottoms' })));
  expect(saveClothingEdits).toHaveBeenCalledWith('owned-item', expect.objectContaining({ category: 'bottom', subCategory: 'bottoms' }));
});

test('saving color without a type change preserves the existing garment subtype', async () => {
  (saveClothingEdits as jest.Mock).mockResolvedValueOnce({ id: 'owned-item' });
  const view = render(<ClothingDetailScreen />);
  fireEvent.press(view.getByLabelText('clothingDetail.editItem'));
  fireEvent.press(view.getByText('clothingEditor.save'));
  await waitFor(() => expect(mockUpdateItem).toHaveBeenCalled());
  expect((saveClothingEdits as jest.Mock).mock.calls[0][1]).not.toHaveProperty('subCategory');
});

test('a save completing after logout cannot update the next account', async () => {
  let finish!: (value: unknown) => void;
  (saveClothingEdits as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const view = render(<ClothingDetailScreen />);
  fireEvent.press(view.getByLabelText('clothingDetail.editItem'));
  fireEvent.press(view.getByText('clothingEditor.save'));
  authEvents.emitLogout(); authEvents.emitLogin('next-owner');
  await act(async () => { finish({ id: 'owned-item' }); });
  expect(mockUpdateItem).not.toHaveBeenCalled();
});


test('a missing save confirmation is treated as failure', async () => {
  const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
  (saveClothingEdits as jest.Mock).mockResolvedValueOnce(null);
  const view = render(<ClothingDetailScreen />);
  fireEvent.press(view.getByLabelText('clothingDetail.editItem'));
  fireEvent.press(view.getByText('clothingEditor.save'));
  await waitFor(() => expect(alert).toHaveBeenCalledWith('common.error', 'clothingEditor.failedToSaveItem'));
  expect(mockUpdateItem).not.toHaveBeenCalled();
  expect(view.getByText('clothingEditor.save')).toBeTruthy();
  alert.mockRestore();
});
