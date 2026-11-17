import React from 'react';
import { Share } from 'react-native';
import { render, fireEvent } from '@testing-library/react-native';
import { BetaFeedbackButton } from '../../components/BetaFeedbackButton';
import { authEvents } from '../../store/authEvents';
import { act } from 'react-test-renderer';
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('expo-constants', () => ({
  __esModule: true,
  default: { expoConfig: { version: '3.0.2' } },
}));
test('feedback requires an explicit helpful choice, reviews before sharing, and discards on account change', () => {
  const share = jest.spyOn(Share, 'share').mockResolvedValue({ action: Share.sharedAction });
  const view = render(
    <BetaFeedbackButton feature="buycheck" diagnostics={{ requestId: 'req-safe' }} />
  );
  fireEvent.press(view.getAllByLabelText('beta.report')[0]);
  expect(view.getByLabelText('beta.review').props.accessibilityState.disabled).toBe(true);
  fireEvent.press(view.getByLabelText('beta.no'));
  fireEvent.changeText(view.getByLabelText('beta.comment'), 'Optional feedback');
  fireEvent.press(view.getByLabelText('beta.review'));
  expect(share).not.toHaveBeenCalled();
  expect(view.getByText(/"requestId": "req-safe"/)).toBeTruthy();
  fireEvent.press(view.getByLabelText('beta.share'));
  expect(share).toHaveBeenCalledTimes(1);
  act(() => authEvents.emitLogin('beta-owner'));
  fireEvent.press(view.getAllByLabelText('beta.report')[0]);
  expect(view.getByLabelText('beta.comment').props.value).toBe('');
  share.mockRestore();
});
