import React from 'react';
import { act, fireEvent, render } from '@testing-library/react-native';
import { View } from 'react-native';
import RootNavigator from '../../navigation/RootNavigator';

const mockAuth = { isAuthenticated: false, isInitialized: true, initializeAuth: jest.fn() };
const mockLanguage = { hasSelectedLanguage: true, isInitialized: true, initializeLanguage: jest.fn() };
const mockUsage = { hydrate: jest.fn() };
const mockPromo = { hydrate: jest.fn(), syncWithServer: jest.fn().mockResolvedValue(undefined) };
const mockSubscription = { initializeSubscription: jest.fn().mockResolvedValue(undefined), verifySubscriptionFromServer: jest.fn() };
const mockReady = jest.fn();
jest.mock('../../store/auth', () => ({ __esModule: true, default: Object.assign((select: any) => select(mockAuth), { getState: () => mockAuth, setState: jest.fn() }), cleanupAuthSubscription: jest.fn(), identifyRestoredPurchases: jest.fn().mockResolvedValue(undefined) }));
jest.mock('../../store/languageStore', () => ({ __esModule: true, default: Object.assign((select: any) => select(mockLanguage), { getState: () => mockLanguage }) }));
jest.mock('../../store/subscriptionStore', () => ({ __esModule: true, default: (select: any) => select(mockSubscription) }));
jest.mock('../../store/dailyUsageStore', () => ({ __esModule: true, default: { getState: () => mockUsage } }));
jest.mock('../../store/promoCodeStore', () => ({ __esModule: true, default: Object.assign((select: any) => select(mockPromo), { getState: () => mockPromo }) }));
jest.mock('../../store/stylePreferenceStore', () => ({ useStylePreferenceStore: (select: any) => select({ hasCompletedOnboarding: true }) }));
jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, default: { getState: () => ({}) } }));
jest.mock('../../src/hooks/useSessionGuard', () => ({ useSessionGuard: jest.fn() }));
jest.mock('../../hooks/useNetworkRecovery', () => ({ useNetworkRecovery: jest.fn() }));
jest.mock('../../src/services/notificationService', () => ({ addNotificationListeners: () => () => {}, notificationService: { initialize: jest.fn().mockResolvedValue(undefined) } }));
jest.mock('../../src/services/iapService', () => ({ iapService: { initialize: jest.fn().mockResolvedValue(undefined) } }));
jest.mock('../../src/services/analyticsService', () => ({ __esModule: true, default: { initialize: jest.fn() } }));
jest.mock('../../navigation/navigationRef', () => ({ navigateTo: jest.fn() }));
jest.mock('../../navigation/liquidTransitions', () => ({ LiquidPresets: {} }));
jest.mock('../../src/utils/perf', () => ({ perfMark: jest.fn(), perfMeasure: jest.fn() }));
jest.mock('../../src/lib/appReady', () => ({ markAppReady: () => mockReady() }));
jest.mock('@react-navigation/native-stack', () => ({ createNativeStackNavigator: () => ({
  Navigator: ({ children }: any) => <>{children}</>, Screen: () => null,
}) }));

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  mockAuth.initializeAuth.mockResolvedValue(undefined);
  mockLanguage.initializeLanguage.mockResolvedValue(undefined);
  mockUsage.hydrate.mockResolvedValue(undefined);
  mockPromo.hydrate.mockResolvedValue(undefined);
});
afterEach(() => jest.useRealTimers());

test('auth starts concurrently with language, but the screen waits for both and local usage', async () => {
  let finishLanguage!: () => void, finishAuth!: () => void, finishUsage!: () => void;
  mockLanguage.initializeLanguage.mockReturnValue(new Promise<void>(resolve => { finishLanguage = resolve; }));
  mockAuth.initializeAuth.mockReturnValue(new Promise<void>(resolve => { finishAuth = resolve; }));
  mockUsage.hydrate.mockReturnValue(new Promise<void>(resolve => { finishUsage = resolve; }));
  const view = render(<RootNavigator />);
  expect(mockAuth.initializeAuth).toHaveBeenCalledTimes(1);
  expect(mockLanguage.initializeLanguage).toHaveBeenCalledTimes(1);
  expect(view.toJSON()).toBeNull();
  await act(async () => finishLanguage());
  expect(mockUsage.hydrate).not.toHaveBeenCalled();
  await act(async () => finishAuth());
  expect(mockUsage.hydrate).toHaveBeenCalledTimes(1);
  expect(view.toJSON()).toBeNull();
  await act(async () => finishUsage());
  expect(mockReady).not.toHaveBeenCalled();
  fireEvent(view.UNSAFE_getByType(View), 'layout');
  expect(mockReady).toHaveBeenCalledTimes(1);
  expect(mockSubscription.initializeSubscription).not.toHaveBeenCalled();
  await act(async () => jest.advanceTimersByTimeAsync(250));
  expect(mockSubscription.initializeSubscription).toHaveBeenCalledTimes(1);
  view.unmount();
});

test('background setup is cancelled if the navigator unmounts first', async () => {
  const view = render(<RootNavigator />);
  await act(async () => {});
  view.unmount();
  await act(async () => jest.runAllTimersAsync());
  expect(mockSubscription.initializeSubscription).not.toHaveBeenCalled();
});

test('an eager idle callback cannot start native services in the opening second', async () => {
  const previousIdle = global.requestIdleCallback;
  global.requestIdleCallback = jest.fn(callback => {
    callback({ didTimeout: false, timeRemaining: () => 16 });
    return 1;
  });
  const { iapService } = require('../../src/services/iapService');
  const { notificationService } = require('../../src/services/notificationService');
  const view = render(<RootNavigator />);
  try {
    await act(async () => {});
    await act(async () => jest.advanceTimersByTimeAsync(1000));
    expect(iapService.initialize).not.toHaveBeenCalled();
    expect(notificationService.initialize).not.toHaveBeenCalled();
    await act(async () => jest.advanceTimersByTimeAsync(250));
    expect(notificationService.initialize).toHaveBeenCalledTimes(1);
    expect(iapService.initialize).not.toHaveBeenCalled();
    await act(async () => jest.advanceTimersByTimeAsync(500));
    expect(iapService.initialize).toHaveBeenCalledTimes(1);
  } finally {
    view.unmount();
    global.requestIdleCallback = previousIdle;
  }
});
