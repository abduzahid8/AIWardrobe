import React from 'react';
import { act, fireEvent, render } from '@testing-library/react-native';
import SignInScreen from '../../screens/SignInScreen';

const mockLogin = jest.fn();
jest.mock('@react-navigation/native', () => ({ useNavigation: () => ({ navigate: jest.fn() }) }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: () => ({ login: mockLogin, signInWithApple: jest.fn(), signInWithGoogle: jest.fn() }) }));
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('../../hooks/useAccessibility', () => ({ useAccessibility: () => ({ isReducedMotionEnabled: true }) }));
jest.mock('react-native-reanimated', () => ({ __esModule: true, default: { View: jest.requireActual('react-native').View }, FadeInDown: {} }));
jest.mock('expo-apple-authentication', () => ({ isAvailableAsync: jest.fn().mockResolvedValue(false) }));
jest.mock('expo-linear-gradient', () => ({ LinearGradient: jest.requireActual('react-native').View }));
jest.mock('react-native-safe-area-context', () => ({ SafeAreaView: jest.requireActual('react-native').View }));
jest.mock('@expo/vector-icons', () => ({ Ionicons: jest.requireActual('react-native').View }));
jest.mock('../../components/ui/ScaledText', () => ({ ScaledText: jest.requireActual('react-native').Text }));

beforeEach(() => { jest.useFakeTimers(); jest.clearAllMocks(); });
afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });
for (const backId of ['signInBackButton', 'changeEmailButton']) {
  test(`${backId} clears a revealed password and masks the next password entry`, async () => {
    const view = render(<SignInScreen />);
    await act(async () => { await Promise.resolve(); });
    fireEvent.changeText(view.getByTestId('emailInputNative'), 'readiness-ui@example.invalid');
    fireEvent.press(view.getByTestId('continueEmailButton'));
    fireEvent.changeText(view.getByTestId('passwordInputNative'), 'local-unsubmitted-fixture');
    fireEvent.press(view.getByLabelText('signIn.showPassword'));
    expect(view.getByTestId('passwordInputNative').props.secureTextEntry).toBe(false);
    fireEvent.press(view.getByTestId(backId));
    expect(view.getByTestId('emailInputNative').props.value).toBe('readiness-ui@example.invalid');
    expect(view.getByTestId('emailInputNative').props.secureTextEntry).not.toBe(true);
    fireEvent.press(view.getByTestId('continueEmailButton'));
    expect(view.getByTestId('passwordInputNative').props.value).toBe('');
    expect(view.getByTestId('passwordInputNative').props.secureTextEntry).toBe(true);
    expect(view.getByLabelText('signIn.showPassword')).toBeTruthy();
    expect(mockLogin).not.toHaveBeenCalled();
    view.unmount();
  });
}
