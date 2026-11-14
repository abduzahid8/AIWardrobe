import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { shoe, owned } from './fixtures';
const mockCloset: any[] = [];
const mockParams = { candidate: shoe, intent: 'check' };
jest.mock('@react-navigation/native', () => ({ useRoute: () => ({ params: mockParams }) }));
jest.mock('../../hooks/useAppNavigation', () => ({
  useAppNavigation: () => ({ goBack: jest.fn(), navigate: jest.fn() }),
}));
jest.mock('../../lib/supabase', () => ({ supabase: { from: jest.fn() } }));
jest.mock('../../store/wardrobeStore', () => ({
  __esModule: true,
  default: (selector: any) => selector({ items: mockCloset }),
}));
jest.mock('../../store/bodyProfileStore', () => ({
  useBodyProfileStore: () => ({ profiles: {}, garmentProfiles: {}, activeProfileId: null }),
}));
jest.mock('../../store/shopCatalogStore', () => ({
  __esModule: true,
  default: (selector: any) => selector({ items: [] }),
}));
jest.mock('../../store/stylePreferenceStore', () => ({
  useStylePreferenceStore: (selector: any) =>
    selector({
      preferences: { avoidColors: [], avoidBrands: [], dislikedPatterns: [], primaryOccasions: [] },
    }),
}));
import Screen from '../../screens/PurchaseIntelligenceScreen';
import { authEvents } from '../../store/authEvents';
import {
  resetPurchaseIntelligence,
  usePurchaseIntelligenceStore as store,
} from '../../store/purchaseIntelligenceStore';
beforeEach(async () => {
  await resetPurchaseIntelligence();
  authEvents.emitLogin('screen-user');
  await store.getState().hydrate();
});
test('BuyCheck is reachable from an exact product and shows uncertainty, saved state and a local-only watch', async () => {
  const view = render(<Screen />);
  fireEvent.press(view.getByText('Run BuyCheck'));
  await waitFor(() => expect(view.getByText('INSUFFICIENT EVIDENCE')).toBeTruthy(), { timeout: 5000 });
  expect(view.getByText(/Recommended size/)).toBeTruthy();
  expect(view.getByText(/unknown shipping, tax, fees/)).toBeTruthy();
  fireEvent.press(view.getByText('Save this product locally'));
  await waitFor(() => expect(store.getState().bookmarks[0].id).toBe(shoe.id), { timeout: 5000 });
  fireEvent.changeText(view.getByLabelText('Watch threshold'), '80');
  fireEvent.press(view.getByText('Watch item price locally'));
  await waitFor(() => expect(store.getState().watches[0].threshold).toBe(80), { timeout: 5000 });
  expect(view.getByText('Saved on this device. Background monitoring is not active.')).toBeTruthy();
});
test('feedback preference is a dismissible screen setting and account changes clear private decisions', async () => {
  const view = render(<Screen />);
  fireEvent(view.getByLabelText('Fit feedback prompts'), 'valueChange', false);
  await waitFor(() => expect(store.getState().feedbackEnabled).toBe(false), { timeout: 5000 });
  fireEvent.press(view.getByText('Run BuyCheck'));
  await waitFor(() => expect(view.queryByText('INSUFFICIENT EVIDENCE')).not.toBeNull(), { timeout: 5000 });
  await act(async () => authEvents.emitLogin('other-user'));
  await waitFor(() => expect(view.queryByText('INSUFFICIENT EVIDENCE')).toBeNull(), { timeout: 5000 });
});

test('a shared URL resolves automatically and cannot leave a previous product verdict selected', async () => {
  const params = mockParams as any;
  params.input = 'https://unknown.test/product';
  params.inputRevision = 1;
  const view = render(<Screen />);
  await waitFor(() => expect(view.queryByText('Run BuyCheck')).toBeNull(), { timeout: 5000 });
  await waitFor(() =>
    expect(view.getByText(/Select a catalog product or enter its title/)).toBeTruthy()
  , { timeout: 5000 });
  delete params.input;
  delete params.inputRevision;
});

test('a decision is hidden as soon as its budget or size inputs change', async () => {
  const view = render(<Screen />);
  fireEvent.press(view.getByText('Run BuyCheck'));
  await waitFor(() => expect(view.queryByText('INSUFFICIENT EVIDENCE')).not.toBeNull(), { timeout: 5000 });
  fireEvent.changeText(view.getByLabelText('Budget'), '50');
  expect(view.queryByText('INSUFFICIENT EVIDENCE')).toBeNull();
});

test('the local decision flow remains usable through Strict Mode effect replay', async () => {
  const view = render(
    <React.StrictMode>
      <Screen />
    </React.StrictMode>
  );
  fireEvent.press(view.getByText('Run BuyCheck'));
  await waitFor(() => expect(view.queryByText('INSUFFICIENT EVIDENCE')).not.toBeNull(), { timeout: 5000 });
});
