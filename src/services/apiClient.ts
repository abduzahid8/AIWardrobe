/**
 * API Client — centralized Axios instance with interceptors.
 *
 * Features:
 * - Auto-attaches Supabase auth token to every request
 * - Retries once on 401 after refreshing the session
 * - Handles 403 / 500 with user-facing alerts
 * - Network error detection + crash breadcrumbs
 */
import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { Alert } from 'react-native';
import useAuthStore from '../../store/auth';
import Config from '../config/env';
import crashReporting from './crashReporting';
import i18n from 'i18next';
import { supabase } from '../../lib/supabase';

// ── Standardized error shape ──
export interface ApiError {
    message: string;
    status: number;
    code?: string;
}

// ── Factory: creates a client with auth + retry for any base URL ──
const createClient = (baseURL: string, timeout: number) => {
    const client = axios.create({
        baseURL,
        timeout,
        headers: { 'Content-Type': 'application/json' },
    });

    // ── Request: auto-attach auth token ──
    client.interceptors.request.use(
        async (config: InternalAxiosRequestConfig) => {
            try {
                const { data: { session } } = await supabase.auth.getSession();
                if (session?.access_token) {
                    config.headers.Authorization = `Bearer ${session.access_token}`;
                }
            } catch {
                // Continue without auth if session check fails
            }
            return config;
        },
        (error) => Promise.reject(error),
    );

    // ── Response: retry on 401, alert on 403/500 ──
    client.interceptors.response.use(
        (response) => response,
        async (error: AxiosError) => {
            const status = error.response?.status;
            const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

            // 401 — try refreshing the session once, then retry
            if (status === 401 && !original._retry) {
                original._retry = true;
                try {
                    const { error: refreshErr } = await supabase.auth.refreshSession();
                    if (!refreshErr) {
                        const { data: { session } } = await supabase.auth.getSession();
                        if (session?.access_token && original.headers) {
                            original.headers.Authorization = `Bearer ${session.access_token}`;
                        }
                        return client(original); // retry with fresh token
                    }
                } catch {
                    // refresh failed — fall through to logout
                }

                // If refresh didn't help, logout
                const { logout } = useAuthStore.getState();
                await logout();
            } else if (status === 403) {
                Alert.alert(
                    i18n.t('api.accessDenied'),
                    i18n.t('api.accessDeniedMessage'),
                );
            } else if (status && status >= 500) {
                crashReporting.logBreadcrumb(`Server error ${status}: ${error.config?.url}`);
                Alert.alert(
                    i18n.t('api.serverError'),
                    i18n.t('api.serverErrorMessage'),
                );
            } else if (!error.response) {
                crashReporting.logBreadcrumb('Network error: no response');
                // Don't alert — NetworkBanner handles this visually
            }

            return Promise.reject(error);
        },
    );

    return client;
};

// ── Exported clients ──

/** Main Express API server (15 s timeout) */
const apiClient = createClient(Config.api.url, 15_000);

/** AliceVision CV microservice (30 s — image processing is slow) */
export const aliceVisionClient = createClient(Config.api.alicevisionUrl, 30_000);

export default apiClient;
