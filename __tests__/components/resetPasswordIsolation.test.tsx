import React from 'react';
import { Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import { act, fireEvent, render, waitFor } from '@testing-library/react-native';
import ResetPasswordScreen from '../../screens/ResetPasswordScreen';
import useAuthStore from '../../store/auth';
import { authEvents } from '../../store/authEvents';
import { createAuthLinking } from '../../src/lib/authLinking';

const mockReset = jest.fn(), mockUpdate = jest.fn(), mockLogout = jest.fn();
jest.mock('../../hooks/useAppNavigation', () => ({ useAppNavigation: () => ({ reset: mockReset, navigate: jest.fn(), canGoBack: () => false }) }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: jest.requireActual('zustand').create(() => ({ session: null })) }));
jest.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
jest.mock('expo-linear-gradient', () => ({ LinearGradient: jest.requireActual('react-native').View }));
jest.mock('react-native-safe-area-context', () => ({ SafeAreaView: jest.requireActual('react-native').View }));
jest.mock('@expo/vector-icons', () => ({ Ionicons: jest.requireActual('react-native').View }));
jest.mock('../../components/ui/ScaledText', () => ({ ScaledText: jest.requireActual('react-native').Text }));

function owner(id: string) {
    authEvents.emitLogin(id);
    useAuthStore.setState({ session: { user: { id } } as any, isAuthenticated: true,
        passwordRecovery: { ownerId: id, revision: authEvents.getRevision() },
        updatePassword: mockUpdate, logout: mockLogout });
}
function form() {
    const view = render(<ResetPasswordScreen />);
    fireEvent.changeText(view.getByPlaceholderText('resetPassword.newPassword'), 'SyntheticPass9');
    fireEvent.changeText(view.getByPlaceholderText('resetPassword.confirmPassword'), 'SyntheticPass9');
    const button = view.getByLabelText('resetPassword.resetPassword');
    return { view, submit: () => fireEvent.press(button) };
}
beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(Haptics.notificationAsync).mockResolvedValue(undefined);
    mockUpdate.mockReset().mockResolvedValue(undefined);
    mockLogout.mockReset().mockImplementation(async () => {
        authEvents.emitLogout(); useAuthStore.setState({ session: null, isAuthenticated: false });
    });
    owner('recovery-owner');
    jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
});
afterEach(() => jest.restoreAllMocks());

it.each(['absent', 'different-owner', 'stale-revision'])('rejects an authenticated session with %s recovery authority', kind => {
    useAuthStore.setState({ passwordRecovery: kind === 'absent' ? null : {
        ownerId: kind === 'different-owner' ? 'other' : 'recovery-owner',
        revision: authEvents.getRevision() - (kind === 'stale-revision' ? 1 : 0),
    } });
    const view = render(<ResetPasswordScreen />);
    expect(view.getByText('resetPassword.invalidSession')).toBeTruthy();
    expect(view.queryByPlaceholderText('resetPassword.newPassword')).toBeNull();
    expect(mockUpdate).not.toHaveBeenCalled();
});

it('sends only one update for rapid repeated submit and signs out before navigating', async () => {
    let finish!: () => void;
    mockUpdate.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    const { submit } = form(); submit(); submit();
    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(mockUpdate).toHaveBeenCalledWith('SyntheticPass9', 'recovery-owner');
    await act(async () => { finish(); });
    expect(Alert.alert).toHaveBeenCalledTimes(1);
    const buttons = jest.mocked(Alert.alert).mock.calls[0][2]!;
    await act(async () => { await buttons[0].onPress!(); });
    expect(mockLogout).toHaveBeenCalledWith('recovery-owner');
    expect(mockReset).toHaveBeenCalledWith({ index: 0, routes: [{ name: 'SignIn' }] });
});

it('invalidates the form when its account changes during the update', async () => {
    let finish!: () => void;
    mockUpdate.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    const { view, submit } = form(); submit();
    act(() => { owner('new-owner'); });
    await act(async () => { finish(); });
    expect(Alert.alert).not.toHaveBeenCalled();
    expect(view.getByText('resetPassword.invalidSession')).toBeTruthy();
    expect(mockLogout).not.toHaveBeenCalled();
});

it('ignores an old success dialog after another account signs in', async () => {
    const { submit } = form(); submit();
    await waitFor(() => expect(Alert.alert).toHaveBeenCalledTimes(1));
    const buttons = jest.mocked(Alert.alert).mock.calls[0][2]!;
    act(() => { owner('new-owner'); });
    await act(async () => { await buttons[0].onPress!(); });
    expect(mockLogout).not.toHaveBeenCalled(); expect(mockReset).not.toHaveBeenCalled();
});

it('does not publish completion after unmount', async () => {
    let finish!: () => void;
    mockUpdate.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    const { view, submit } = form(); submit(); view.unmount();
    await act(async () => { finish(); });
    expect(Alert.alert).not.toHaveBeenCalled();
});

it.each(['cold', 'warm'])('binds the reset form to the recovered account after %s link processing', async flow => {
    let finish!: () => void;
    const work = new Promise<void>(resolve => { finish = resolve; });
    let receive!: (event: { url: string }) => void;
    const url = 'com.aiwardrobe://reset-password?code=synthetic';
    const linking = createAuthLinking({
        getInitialURL: async () => url,
        addEventListener: (_type, listener) => { receive = listener; return { remove: jest.fn() }; },
    }, async () => { await work; owner('recovered-account'); });
    const navigate = jest.fn();
    let pending: Promise<unknown> | undefined;
    const unsubscribe = flow === 'warm' ? linking.subscribe(navigate) : undefined;
    if (flow === 'warm') receive({ url });
    else pending = linking.getInitialURL().then(navigate);
    await act(async () => { await Promise.resolve(); });
    expect(navigate).not.toHaveBeenCalled();
    await act(async () => { finish(); await pending; });
    await waitFor(() => expect(navigate).toHaveBeenCalledWith(url));
    const { submit } = form(); submit();
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledWith('SyntheticPass9', 'recovered-account'));
    unsubscribe?.();
});
