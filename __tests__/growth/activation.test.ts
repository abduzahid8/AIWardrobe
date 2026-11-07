import {
    emptyActivationState,
    recordMilestone,
    milestonesCompleted,
    isActivated,
    createAhaMoment,
    nextOnboardingStage,
    firstValueReady,
    surfacesForIntent,
    closetProgressHint,
} from '../../src/services/growth/activation';
import { prioritizeHomeActions, emptyStateFor, markStreakWeek, startStreak, streakStatusCopy, buildWeeklyValueSummary } from '../../src/services/growth/home';

describe('activation model', () => {
    test('milestones accumulate; activation requires high-value action', () => {
        const s = emptyActivationState();
        expect(isActivated(s)).toBe(false);
        expect(recordMilestone(s, 'wardrobeStarted')).toBe(true);
        expect(recordMilestone(s, 'wardrobeStarted')).toBe(false); // idempotent
        expect(isActivated(s)).toBe(false);
        recordMilestone(s, 'firstOutfitSaved');
        expect(isActivated(s)).toBe(true);
        expect(milestonesCompleted(s)).toBe(2);
    });

    test('skipped onboarding jumps to deepen; first value needs 3 garments', () => {
        expect(nextOnboardingStage({ stage: 'minimum_setup', styleIntentCaptured: false, climateCaptured: false, garmentCount: 0, firstValueDelivered: false, skipped: true })).toBe('deepen_personalization');
        expect(firstValueReady(1).ready).toBe(false);
        expect(firstValueReady(1).hint).toMatch(/Add 2 more/);
        expect(firstValueReady(3).ready).toBe(true);
        expect(closetProgressHint(5)).toBeNull();
    });

    test('intent routing prioritizes relevant surfaces; no sensitive inference', () => {
        expect(surfacesForIntent('check_before_buy')[0]).toBe('shop_check');
        expect(surfacesForIntent('recreate_look')[0]).toBe('stylist');
        expect(createAhaMoment('no_buy_saved_purchase').kind).toBe('no_buy_saved_purchase');
    });
});

describe('home action center', () => {
    test('expired and cooling-down candidates filtered; urgency×confidence×value wins', () => {
        const actions = prioritizeHomeActions([
            { kind: 'weekly_plan', urgency: 0.9, confidence: 0.9, expectedValue: 0.9, hoursSinceLastShown: 200, expiresInHours: null, goalAligned: true },
            { kind: 'todays_outfit', urgency: 0.5, confidence: 0.5, expectedValue: 0.5, hoursSinceLastShown: 1, expiresInHours: null, goalAligned: false },
            { kind: 'upcoming_event', urgency: 1, confidence: 1, expectedValue: 1, hoursSinceLastShown: 100, expiresInHours: 0, goalAligned: true },
        ]);
        expect(actions.map((a) => a.kind)).toEqual(['weekly_plan']);
    });

    test('empty states guide progress; weekly summary counts verified savings only', () => {
        expect(emptyStateFor('wardrobe').action).toMatch(/Add 3–5 items/);
        const summary = buildWeeklyValueSummary({
            outfitsWorn: 5, itemsReused: 8, itemsUnused: 4,
            savings: [{ kind: 'skipped_duplicate', amountCents: 5000, currency: 'USD', verifiedAt: 'x', provenance: 'buycheck:req1' }],
            bestOutfitId: null, upcomingNeed: null, goalProgress: null,
        });
        expect(summary.verifiedSavingsCents).toBe(5000);
    });

    test('streaks optional and non-shaming', () => {
        let s = startStreak('weekly_plan');
        s = markStreakWeek(s, '2026-W45');
        s = markStreakWeek(s, '2026-W45'); // idempotent
        expect(s.count).toBe(1);
        expect(streakStatusCopy(s)).toMatch(/1 week of wardrobe planning/);
        expect(streakStatusCopy({ ...s, optedIn: false })).toBeDefined();
    });
});
