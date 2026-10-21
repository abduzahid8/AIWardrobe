import { useEffect } from 'react';
import { AppState } from 'react-native';
import useSubscriptionStore, { FREE_TRIAL_DAYS } from '../../store/subscriptionStore';

/** Keep access and countdowns current even when no subscription screen is open. */
export function useSubscriptionClock(): void {
    useEffect(() => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        let stopped = false;
        const schedule = () => {
            if (timer !== undefined) clearTimeout(timer);
            if (stopped) return;
            const state = useSubscriptionStore.getState();
            const deadlines = [
                state.tier !== 'free' && state.expiryDate ? Date.parse(state.expiryDate) : NaN,
                state.isTrialActive && state.trialStartedAt
                    ? Date.parse(state.trialStartedAt) + FREE_TRIAL_DAYS * 86400000 : NaN,
            ].filter(end => Number.isFinite(end) && end > Date.now());
            const remaining = deadlines.length ? Math.min(...deadlines) - Date.now() : NaN;
            // A short final timeout reconciles exactly at an access deadline; minute ticks
            // update days and handle device-clock changes without network traffic.
            const delay = Number.isFinite(remaining) && remaining > 0
                ? Math.min(60_000, remaining) : 60_000;
            timer = setTimeout(tick, delay);
        };
        const tick = () => {
            if (stopped) return;
            useSubscriptionStore.getState().refreshTimeState();
            schedule();
        };
        const unsubscribe = useSubscriptionStore.subscribe(schedule);
        const appState = AppState.addEventListener('change', state => {
            if (state === 'active') tick();
        });
        tick();
        return () => {
            stopped = true;
            if (timer !== undefined) clearTimeout(timer);
            unsubscribe();
            appState.remove();
        };
    }, []);
}
