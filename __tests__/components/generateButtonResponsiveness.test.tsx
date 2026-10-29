import React from 'react';
import { fireEvent, render } from '@testing-library/react-native';
import * as Haptics from 'expo-haptics';
import GenerateOutfitButton from '../../features/outfit-generator/components/GenerateOutfitButton';

jest.mock('expo-haptics', () => ({ impactAsync: jest.fn(), ImpactFeedbackStyle: { Medium: 'medium' } }));

test('opens the feature immediately while haptics are unresolved', async () => {
  let fail!: (error: Error) => void;
  (Haptics.impactAsync as jest.Mock).mockReturnValue(new Promise((_, reject) => { fail = reject; }));
  const onPress = jest.fn(), item = { _id: 'shirt' };
  const view = render(<GenerateOutfitButton item={item} onPress={onPress} accessibilityLabel="Create outfit" />);
  fireEvent.press(view.getByLabelText('Create outfit'));
  expect(onPress).toHaveBeenCalledWith(item);
  fail(new Error('Haptics unavailable'));
  await Promise.resolve();
  expect(onPress).toHaveBeenCalledTimes(1);
});
