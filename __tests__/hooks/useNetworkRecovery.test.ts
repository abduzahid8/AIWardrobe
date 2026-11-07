import { act, renderHook } from '@testing-library/react-native';
import NetInfo from '@react-native-community/netinfo';
import { useNetworkRecovery } from '../../hooks/useNetworkRecovery';

jest.mock('@react-native-community/netinfo', () => ({ __esModule: true, default: { addEventListener: jest.fn() } }));
test('recovers once after confirmed offline-to-online transition and removes listener', () => {
  let notify: (state: any) => void = () => {};
  const unsubscribe = jest.fn();
  (NetInfo.addEventListener as jest.Mock).mockImplementation((callback) => { notify = callback; return unsubscribe; });
  const retry = jest.fn();
  const hook = renderHook(() => useNetworkRecovery(retry));
  act(() => notify({ isConnected: true, isInternetReachable: true }));
  expect(retry).not.toHaveBeenCalled();
  act(() => notify({ isConnected: false, isInternetReachable: false }));
  act(() => notify({ isConnected: true, isInternetReachable: null }));
  expect(retry).not.toHaveBeenCalled();
  act(() => notify({ isConnected: true, isInternetReachable: true }));
  act(() => notify({ isConnected: true, isInternetReachable: true }));
  expect(retry).toHaveBeenCalledTimes(1);
  hook.unmount();
  expect(unsubscribe).toHaveBeenCalledTimes(1);
});
