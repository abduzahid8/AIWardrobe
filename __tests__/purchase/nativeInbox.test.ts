import { NativeModules, AppState } from 'react-native';
const mockBridge = { setOwner: jest.fn(), consume: jest.fn() };
Object.defineProperty(NativeModules, 'PurchaseShareInbox', {
  value: mockBridge,
  configurable: true,
});
const mockSubscribe = jest.spyOn(AppState, 'addEventListener');
import { authEvents } from '../../store/authEvents';
import { listenForPurchaseInbox } from '../../src/services/purchase/nativeShareInbox';
beforeEach(() => {
  jest.clearAllMocks();
  authEvents.emitLogout();
  mockSubscribe.mockReturnValue({ remove: jest.fn() });
  mockBridge.consume.mockResolvedValue(null);
});
test('an unknown startup owner never clears an inbox before authentication can restore it', async () => {
  const receive = jest.fn(),
    stop = listenForPurchaseInbox(receive);
  expect(mockBridge.setOwner).not.toHaveBeenCalled();
  mockBridge.consume.mockResolvedValue('https://merchant.test/product');
  authEvents.emitLogin('restored-owner');
  await Promise.resolve();
  expect(mockBridge.setOwner).toHaveBeenCalledWith('restored-owner');
  expect(receive).toHaveBeenCalledWith('https://merchant.test/product');
  stop();
});
test('a delayed native payload cannot be delivered after an account transition', async () => {
  authEvents.emitLogin('first');
  let finish!: (value: string) => void;
  mockBridge.consume.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      })
  );
  const receive = jest.fn(),
    stop = listenForPurchaseInbox(receive);
  authEvents.emitLogout();
  authEvents.emitLogin('second');
  finish('https://merchant.test/private-item');
  await Promise.resolve();
  expect(receive).not.toHaveBeenCalled();
  expect(mockBridge.setOwner).toHaveBeenCalledWith('');
  stop();
});
