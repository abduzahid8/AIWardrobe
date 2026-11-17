import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { FitFeedback } from '../../components/purchase/FitFeedback';
import { authEvents } from '../../store/authEvents';
import {
  usePurchaseIntelligenceStore as store,
  resetPurchaseIntelligence,
} from '../../store/purchaseIntelligenceStore';
beforeEach(async () => {
  await resetPurchaseIntelligence();
  authEvents.emitLogin('feedback-user');
  await store.getState().hydrate();
});
test('one tap persists immediately; zones and reference size remain optional', async () => {
  const view = render(<FitFeedback productId="garment" brand="Brand" category="top" />);
  fireEvent.press(view.getByText('Perfect'));
  await waitFor(() => expect(store.getState().evidence).toHaveLength(1));
  expect(store.getState().evidence[0].zones).toBeUndefined();
  fireEvent.press(view.getByText('Add optional details'));
  fireEvent.press(view.getByText('shoulders'));
  fireEvent.press(view.getByText('tight'));
  await waitFor(() => expect(store.getState().evidence[0].zones?.shoulders).toBe('tight'));
  fireEvent.changeText(view.getByLabelText('Garment size'), 'M');
  fireEvent.press(view.getByText('Use as fit reference'));
  await waitFor(() =>
    expect(store.getState().evidence[0]).toMatchObject({ reference: true, sizeLabel: 'M' })
  );
});
test('Not now dismisses the contextual prompt and disabled setting hides it', async () => {
  const view = render(<FitFeedback productId="garment" category="top" />);
  fireEvent.press(view.getByText('Not now'));
  await waitFor(() => expect(view.queryByText('How does this fit?')).toBeNull());
  view.rerender(<FitFeedback productId="other" category="top" />);
  await waitFor(() => expect(view.queryByText('How does this fit?')).not.toBeNull());
  await act(async () => store.getState().setFeedbackEnabled(false));
  expect(view.queryByText('How does this fit?')).toBeNull();
});

test('an old wardrobe item cannot show or edit fit evidence after an account switch', async () => {
  const view = render(
    <FitFeedback ownerId="feedback-user" productId="owned-private" brand="Brand" category="top" />
  );
  fireEvent.press(view.getByText('Perfect'));
  await waitFor(() => expect(store.getState().evidence).toHaveLength(1));
  await act(async () => authEvents.emitLogin('different-owner'));
  expect(view.queryByText('Garment size')).toBeNull();
  expect(view.queryByLabelText('Garment size')).toBeNull();
});
