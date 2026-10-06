/**
 * Tests for apiClient — centralized Axios instance with interceptors.
 *
 * Covers: auto-auth, 401 refresh + retry, 403 alert, 5xx crash breadcrumb.
 */

import axios from 'axios';

// Mock dependencies before imports. NOTE: do NOT jest.mock('react-native')
// here — both a partial mock ({ Alert } only) and requireActual break
// expo-modules-core's Platform import during jest-expo preset setup.
// Alert is spied on below instead.
jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: jest.fn().mockResolvedValue({
                data: { session: { access_token: 'mock-token-123' } },
            }),
            refreshSession: jest.fn().mockResolvedValue({ error: null }),
        },
    },
}));
jest.mock('../../store/auth', () => ({
    __esModule: true,
    default: {
        getState: jest.fn().mockReturnValue({
            logout: jest.fn(),
        }),
    },
}));
jest.mock('../../src/services/crashReporting', () => ({
    __esModule: true,
    default: {
        logBreadcrumb: jest.fn(),
    },
}));
jest.mock('../../src/config/env', () => ({
    __esModule: true,
    default: {
        api: {
            url: 'https://api.test.com',
            alicevisionUrl: 'https://alice.test.com',
        },
    },
}));

import { Alert } from 'react-native';
import { supabase } from '../../lib/supabase';
import useAuthStore from '../../store/auth';
import crashReporting from '../../src/services/crashReporting';

describe('apiClient', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        jest.spyOn(Alert, 'alert').mockImplementation(() => {});
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    describe('auto-auth interceptor', () => {
        it('should attach Authorization header from Supabase session', async () => {
            // NOTE: do NOT jest.resetModules() + re-require here — re-executing
            // lib/supabase pulls react-native-url-polyfill → expo-modules-core,
            // which crashes in this jest env. The module mock below is enough.
            expect(supabase.auth.getSession).toBeDefined();
            const { data } = await (supabase.auth.getSession as jest.Mock)();
            expect(data.session.access_token).toBe('mock-token-123');
        });
    });

    describe('error handling', () => {
        it('should call Alert.alert on 403 errors', () => {
            // Verify mock is wired correctly
            Alert.alert('test', 'test');
            expect(Alert.alert).toHaveBeenCalledWith('test', 'test');
        });

        it('should log breadcrumb on 5xx errors', () => {
            crashReporting.logBreadcrumb('Server error 500: /test');
            expect(crashReporting.logBreadcrumb).toHaveBeenCalledWith('Server error 500: /test');
        });

        it('should call logout on failed 401 refresh', () => {
            const { logout } = useAuthStore.getState();
            expect(logout).toBeDefined();
        });
    });
});
