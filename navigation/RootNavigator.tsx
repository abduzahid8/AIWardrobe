import React, { useEffect, useState, useMemo, Suspense } from "react";
import * as SplashScreen from "expo-splash-screen";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { View } from "react-native";

// Eager: only the auth shell + tab host needed for first frame.
// Everything else is lazy so the startup JS bundle stays small and
// cold start doesn't evaluate camera/video/AI/photo-picker native modules
// until the user actually navigates there (fixes 4/10 freeze on launch).
import SignInScreen from "../screens/SignInScreen";
import SignUpScreen from "../screens/SignUpScreen";
import TabNavigator from "../navigation/TabNavigator";
import ForgotPasswordScreen from "../screens/ForgotPasswordScreen";
import ResetPasswordScreen from "../screens/ResetPasswordScreen";
import AuthCallbackScreen from "../screens/AuthCallbackScreen";
import LanguageSelectionScreen from "../screens/LanguageSelectionScreen";

// Lazy screen helper — Suspense with blank fallback (navigator already
// shows the previous screen during the split-second load).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const lazyScreen = (loader: () => Promise<{ default: React.ComponentType<any> }>) => {
  const LazyComp = React.lazy(loader);
  const Wrapped = (props: Record<string, unknown>) => (
    <Suspense fallback={<View style={{ flex: 1, backgroundColor: "#FFFFFF" }} />}>
      <LazyComp {...(props as object)} />
    </Suspense>
  );
  return Wrapped;
};

const ScanWardrobeScreen = lazyScreen(() => import("../screens/ScreenWardrobe"));
const AIOutfitmaker = lazyScreen(() => import("../screens/AIOutfitmaker"));
const WardrobeVideoScreen = lazyScreen(() => import("../screens/WardrobeVideoScreen"));
const CameraScreen = lazyScreen(() => import("../screens/CameraScreen"));
const ReviewScreen = lazyScreen(() => import("../screens/ReviewScreen"));
const OutfitCalendarScreen = lazyScreen(() => import("../screens/OutfitCalendarScreen"));
const PaywallScreen = lazyScreen(() => import("../screens/PaywallScreen"));
const OutfitInspoScreen = lazyScreen(() => import("../screens/OutfitInspoScreen"));
const CreateAvatarScreen = lazyScreen(() => import("../screens/CreateAvatarScreen"));
const MyClosetScreen = lazyScreen(() => import("../screens/MyClosetScreen"));
const WeeklyInsightsScreen = lazyScreen(() => import("../screens/WeeklyInsightsScreen"));
const PrivacyPolicyScreen = lazyScreen(() => import("../screens/PrivacyPolicyScreen"));
const TermsOfServiceScreen = lazyScreen(() => import("../screens/TermsOfServiceScreen"));
const ClothingDetailEditor = lazyScreen(() => import("../components/ClothingDetailEditor"));
const ClothingDetailScreen = lazyScreen(() => import("../screens/ClothingDetailScreen"));
const TrialExpiredScreen = lazyScreen(() => import("../screens/TrialExpiredScreen"));
const PromoCodeScreen = lazyScreen(() => import("../screens/PromoCodeScreen"));
const AdminPanelScreen = lazyScreen(() => import("../screens/AdminPanelScreen"));
const GuideScreen = lazyScreen(() => import("../screens/GuideScreen"));
const StyleQuizScreen = lazyScreen(() => import("../screens/StyleQuizScreen"));
const BodyProfileScreen = lazyScreen(() => import("../screens/BodyProfileScreen"));
import useAuthStore, { cleanupAuthSubscription } from "../store/auth";
import useSubscriptionStore from "../store/subscriptionStore";
import useDailyUsageStore from "../store/dailyUsageStore";
import useWardrobeStore from "../store/wardrobeStore";
import { addNotificationListeners } from "../src/services/notificationService";
import { notificationService } from "../src/services/notificationService";
import { RootStackParamList } from "./types";
import { navigationRef, navigateTo } from "./navigationRef";
import { useSessionGuard } from "../src/hooks/useSessionGuard";
import analyticsService from "../src/services/analyticsService";
import { iapService } from "../src/services/iapService";
import { LiquidPresets } from "./liquidTransitions";
import usePromoCodeStore from "../store/promoCodeStore";
import { useStylePreferenceStore } from "../store/stylePreferenceStore";
import useLanguageStore from "../store/languageStore";
import { perfMark, perfMeasure } from "../src/utils/perf";


const Stack = createNativeStackNavigator<RootStackParamList>();

const RootNavigator = () => {
  useSessionGuard();

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isInitialized = useAuthStore((s) => s.isInitialized);
  const [isAppReady, setIsAppReady] = useState(false);
  const initializeSubscription = useSubscriptionStore((s) => s.initializeSubscription);
  const verifySubscriptionFromServer = useSubscriptionStore((s) => s.verifySubscriptionFromServer);
  const hydratePromoCodeStore = usePromoCodeStore((s) => s.hydrate);
  const hydratePromoCodeStoreSync = () => {
    usePromoCodeStore.getState().syncWithServer?.().catch(() => {});
  };
  const hasCompletedOnboarding = useStylePreferenceStore((s) => s.hasCompletedOnboarding);
  const onboardingStep = useStylePreferenceStore((s) => s.onboardingStep);
  const hasSelectedLanguage = useLanguageStore((s) => s.hasSelectedLanguage);
  const isLanguageInitialized = useLanguageStore((s) => s.isInitialized);

  // NOTE: expired trials NEVER block the whole app. When the 7-day trial
  // ends the user keeps free-tier access (see TIER_FEATURES.free) and each
  // premium feature redirects to Paywall individually via useSubscriptionGate.
  // The TrialExpired screen stays registered below for optional upsell
  // navigation, but it is no longer used as a routing gate.

  // NOTE: promo gate is intentionally disabled (no const needed).
  // Apple Guideline 2.1(a) requires that users can access the free tier
  // without being forced through a promo code or paywall. The PromoCode
  // screen remains accessible voluntarily (e.g. from Profile screen).
  const showOnboardingGate = useMemo(
    () => isAuthenticated && !hasCompletedOnboarding && onboardingStep < 6,
    [isAuthenticated, hasCompletedOnboarding, onboardingStep],
  );
  const showOnboardingPaywall = useMemo(
    () => isAuthenticated && !hasCompletedOnboarding && onboardingStep >= 6,
    [isAuthenticated, hasCompletedOnboarding, onboardingStep],
  );

  useEffect(() => {
    const initialize = async () => {
      try {
      perfMark('app:init');
      // 0. Language first — local AsyncStorage read only (<50ms). Must
      // resolve before routing so the first-run language gate is correct.
      // bootstrapStores() also calls this; the store guards double-hydration.
      await useLanguageStore.getState().initializeLanguage().catch((err) =>
        console.warn('[RootNavigator] initializeLanguage failed', err),
      );
      // 1. Auth must happen first — every other service wants to know who
      // the user is before initializing (for user-scoped analytics, IAP, etc.)
      // Hard 5-second timeout so the app never stays frozen on the splash if
      // Supabase's getSession() hangs (e.g. first cold-start on slow networks).
      const { initializeAuth } = useAuthStore.getState();
      await Promise.race([
        initializeAuth(),
        new Promise<void>((_, reject) =>
          setTimeout(() => reject(new Error('initializeAuth timed out after 5000ms')), 5000)
        ),
      ]).catch((err) => {
        console.warn('[RootNavigator] initializeAuth timeout or error — forcing isInitialized', err);
        // Ensure isInitialized is always set so the app can render
        useAuthStore.setState({ isInitialized: true, loading: false });
      });
      perfMeasure('app:init');

      // 2. Analytics is synchronous — start it immediately so early events
      // (like "app_opened") land in the queue.
      analyticsService.initialize();

      // 3. Everything else is independent of each other and of the render
      // path. Run them in parallel and never block the UI on them.
      // Individual failures are swallowed so one slow service can't
      // block the others.
      //
      // rehydrateFromCloud is included here so wardrobe data fetching
      // starts immediately after auth resolves, in parallel with the other
      // services, instead of waiting for all five to finish first
      // (fixes Defect 1.3 — was previously called sequentially after this block).
      const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();
      // Start background services AFTER the first frame so they never
      // compete with splash-hide + initial render for JS thread / network.
      // Previously notificationService (native permission dialogs + push
      // token network call), RevenueCat configure, and wardrobe cloud sync
      // all fired in the same tick as auth resolution — the thundering herd
      // that stalled cold start. Now:
      //  - wardrobe rehydrate starts immediately (longest pole, outfits need it)
      //  - notifications + IAP init are deferred ~1s to idle
      //  - promo/subscription server reconciliation runs after app-ready
      const deferToIdle = (fn: () => void, ms = 1000) => {
        if (typeof (global as any).requestIdleCallback === 'function') {
          (global as any).requestIdleCallback(fn, { timeout: ms + 2000 });
        } else {
          setTimeout(fn, ms);
        }
      };
      deferToIdle(() => {
        notificationService.initialize().catch((err) =>
          console.warn('[RootNavigator] notificationService failed', err),
        );
      });
      deferToIdle(() => {
        iapService.initialize().catch((err) =>
          console.warn('[RootNavigator] iapService failed', err),
        );
      }, 1500);
      if (authStatus && currentUser?.id) {
        useWardrobeStore.getState().rehydrateFromCloud(false)
          .catch((err) => {
            if (__DEV__) console.warn('[RootNavigator] rehydrateFromCloud failed', err);
          });
      }

      // initializeSubscription is NOT included in the race below — when there's
      // no locally-cached trial date it deliberately awaits a server round trip
      // (see subscriptionStore.ts) to avoid flashing an incorrect "trial expired"
      // screen, so its runtime isn't bounded by a fast local-storage budget. It
      // keeps `isPending` true in the store until it resolves, which is what
      // actually protects the UI, so app-ready doesn't need to wait for it too.
      initializeSubscription().catch((err) =>
        console.warn('[RootNavigator] initializeSubscription failed', err),
      );

      // We only await fast local AsyncStorage hydrations (daily usage, promo codes).
      // Both are local-only now (<50ms typical) — promo server reconciliation
      // moved to syncWithServer() after first frame. The timeout is a pure
      // safety net and should never fire; if it does, something is blocking
      // the JS thread.
      const INIT_TIMEOUT_MS = 800;
      const timeoutPromise = new Promise<void>((_, reject) =>
        setTimeout(() => reject(new Error(`App initialization timed out after ${INIT_TIMEOUT_MS}ms`)), INIT_TIMEOUT_MS)
      );

      await Promise.race([
        Promise.all([
          useDailyUsageStore.getState().hydrate().catch((err) =>
            console.warn('[RootNavigator] dailyUsage hydrate failed', err),
          ),
          hydratePromoCodeStore().catch((err) =>
            console.warn('[RootNavigator] promoCode hydrate failed', err),
          ),
        ]),
        timeoutPromise,
      ]).catch((err) => {
        console.error('[RootNavigator] Initialization timeout or error — forcing app ready', err);
      });

      // 4. Identify the user once services are ready. Verification against
      // the server is fire-and-forget — the UI doesn't need to block on it.
      // Deferred to idle so it never competes with first render.
      if (authStatus && currentUser?.id) {
        analyticsService.setUserId(currentUser.id);
        deferToIdle(() => {
          iapService.identify(currentUser.id).catch(() => {});
          verifySubscriptionFromServer().catch((err) =>
            console.warn('[RootNavigator] verifySubscriptionFromServer failed', err),
          );
          hydratePromoCodeStoreSync();
        }, 2000);
      } else {
        // Logged-out: still reconcile promo cache in background (cheap, local-first).
        deferToIdle(() => hydratePromoCodeStoreSync(), 2000);
      }

      } catch (fatalErr) {
        // If auth or any unguarded step throws, we still need to let the
        // app render so the user sees something instead of a permanent
        // splash screen (which iOS eventually kills → "crash on launch").
        console.error('[RootNavigator] Fatal init error — forcing app ready', fatalErr);
      }

      // 5. Mark app as fully ready — splash screen can now be hidden safely.
      // This MUST always run, even if init failed, so the user isn't stuck
      // on a blank splash that iOS terminates.
      setIsAppReady(true);
    };

    initialize();

    let removeListeners = () => {};
    try {
      removeListeners = addNotificationListeners(
        undefined,
        (response) => {
          const screen = response.notification.request.content.data?.screen;
          if (screen) {
            navigateTo(screen as keyof RootStackParamList);
          }
        }
      );
    } catch (e) {
      console.warn('[RootNavigator] Failed to add notification listeners', e);
    }

    return () => {
      try {
        removeListeners();
      } catch (e) {
        console.warn('[RootNavigator] Failed to remove notification listeners', e);
      }
      cleanupAuthSubscription();
    };
  }, []);

  // Gates are now rendered as the initial route (see stack below) instead
  // of being pushed via navigateTo side-effects. Declarative routing is
  // race-free: when `showPromoGate` / onboarding gates flip back to false,
  // React Navigation unmounts the gate automatically.
  // Expired trials are intentionally NOT a gate — see note above.

  useEffect(() => {
    if (isAppReady && isInitialized && isLanguageInitialized) {
      // Hide the splash screen only after ALL services are initialized and the app is ready to render.
      // Defer slightly to ensure the Navigator has mounted and rendered its first frame.
      const timer = setTimeout(() => {
        SplashScreen.hideAsync().catch(() => undefined);
      }, 100);

      // Warm non-Home tab caches (Inspo catalog/capsules/hero, Closet + Inspo
      // images) on idle so the first visit to each tab finds warm caches
      // instead of freezing on cold queries + image decode. Deferred well
      // past first frame; staggered inside the service; never throws.
      const prewarmTimer = setTimeout(() => {
        import('../src/services/tabPrewarmService').then(({ prewarmTabs }) => {
          prewarmTabs().catch(() => {});
        }).catch(() => {});
      }, 2500);

      return () => {
        clearTimeout(timer);
        clearTimeout(prewarmTimer);
      };
    }
  }, [isAppReady, isInitialized, isLanguageInitialized]);

  if (!isInitialized || !isAppReady || !isLanguageInitialized) {
    // Keep the splash screen visible and render nothing while initializing.
    return null;
  }

  // NOTE: the first-run language gate lives INSIDE the main navigator
  // (LanguageSelection as a pushable screen of the logged-out branch), NOT
  // as a separate early-return navigator. A separate navigator would
  // remount on selection and destroy the back stack — keeping one stable
  // navigator is what makes iOS edge-swipe / Android back from SignIn
  // natively return to the language chooser.

  // Logged-out initial route: fresh installs (no saved @app_language)
  // start on the chooser; everyone else starts on SignIn. Evaluated at
  // mount — the splash gate above guarantees language hydration ran first.
  // Logged-in mounts always start on Main (first authed child).
  const loggedOutInitialRoute: keyof RootStackParamList = hasSelectedLanguage
    ? 'SignIn'
    : 'LanguageSelection';

  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
        // True platform transitions (UIViewController / Fragment) — pushes,
        // pops and edge-swipes run on the native side at 60fps. Native
        // screens detach by default, so no detach prop is needed here.
        ...LiquidPresets.slide,
        contentStyle: {
          // Always use the light background — all screens in this app use the
          // light design system. Using `colors.background` caused a navy blue
          // (#0A1931) flash between tab switches when the device is in dark mode
          // because `colors` is resolved at module load time from the system
          // color scheme.
          backgroundColor: '#FFFFFF',
        },
      }}
      initialRouteName={isAuthenticated ? undefined : loggedOutInitialRoute}
    >
      {isAuthenticated ? (
        showOnboardingGate ? (
          <Stack.Screen
            name="StyleQuiz"
            component={StyleQuizScreen}
            options={{ ...LiquidPresets.fade, gestureEnabled: false }}
          />
        ) : showOnboardingPaywall ? (
          <Stack.Screen
            name="Paywall"
            component={PaywallScreen}
            options={{
              ...LiquidPresets.fade,
              gestureEnabled: false,
            }}
          />
        ) : (
          <>
            <Stack.Screen name="Main" component={TabNavigator} />

            {/* Main tab navigation with Home, Add, and Profile */}
            {/* <Stack.Screen name="Home" component={TabNavigator} /> */}

            {/* Important: name should match ParamList */}
            <Stack.Screen
              name="ReviewScan"          // Route name (for navigation.navigate)
              component={ReviewScreen}   // Component itself (from file)
              options={{
                headerShown: false,
                ...LiquidPresets.rise,
              }}
            />

            <Stack.Screen
              name="AIOutfit"
              component={AIOutfitmaker}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="OutfitInspo"
              component={OutfitInspoScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="CreateAvatar"
              component={CreateAvatarScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="ScanWardrobe"
              component={ScanWardrobeScreen}
              options={{ ...LiquidPresets.rise }}
            />
            <Stack.Screen
              name="WardrobeVideo"
              component={WardrobeVideoScreen}
              options={{
                ...LiquidPresets.rise,
              }}
            />
            <Stack.Screen
              name="Camera"
              component={CameraScreen}
              options={{
                ...LiquidPresets.rise,
                headerShown: false,
              }}
            />
            <Stack.Screen
              name="Calendar"
              component={OutfitCalendarScreen}
              options={{ ...LiquidPresets.slide }}
            />

            <Stack.Screen
              name="MyCloset"
              component={MyClosetScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="WeeklyInsights"
              component={WeeklyInsightsScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="PrivacyPolicy"
              component={PrivacyPolicyScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="TermsOfService"
              component={TermsOfServiceScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />
            <Stack.Screen
              name="ClothingDetailEditor"
              component={ClothingDetailEditor}
              options={{
                ...LiquidPresets.rise,
              }}
            />
            <Stack.Screen
              name="ClothingDetail"
              component={ClothingDetailScreen}
              options={{
                ...LiquidPresets.slide,
              }}
            />

            {/* Global Paywall */}
            <Stack.Screen
              name="Paywall"
              component={PaywallScreen}
              options={{
                ...LiquidPresets.fade,
                presentation: 'card',
                gestureEnabled: false,
              }}
            />

            {/* Trial Expired (always available in stack for deep links / manual nav) */}
            <Stack.Screen
              name="TrialExpired"
              component={TrialExpiredScreen}
              options={{ ...LiquidPresets.fade }}
            />
            <Stack.Screen
              name="PromoCode"
              component={PromoCodeScreen}
              options={{ ...LiquidPresets.fade, gestureEnabled: false }}
            />

            {/* Admin Panel — shop catalog management for admin users */}
            <Stack.Screen
              name="AdminPanel"
              component={AdminPanelScreen}
              options={{ ...LiquidPresets.rise }}
            />

            {/* Guide — editable onboarding/guide page */}
            <Stack.Screen
              name="Guide"
              component={GuideScreen}
              options={{ ...LiquidPresets.slide }}
            />
            <Stack.Screen
              name="BodyProfile"
              component={BodyProfileScreen}
              options={{ ...LiquidPresets.slide }}
            />
          </>
        )
        ) : (
        <>
          {/*
            Language chooser — first screen on fresh installs
            (initialRouteName), otherwise only reachable via native back
            (iOS edge-swipe / Android back) from SignIn. Gestures are off
            here: it is a root, there is nothing to pop to.
          */}
          <Stack.Screen
            name="LanguageSelection"
            component={LanguageSelectionScreen}
            options={{ ...LiquidPresets.fade, gestureEnabled: false }}
          />
          <Stack.Screen
            name="SignIn"
            component={SignInScreen}
            // Fade visuals, but swipe-back must stay ON so the
            // language chooser underneath remains natively reachable.
            options={{ ...LiquidPresets.fade, gestureEnabled: true, gestureDirection: 'horizontal' }}
          />
          <Stack.Screen
            name="SignUp"
            component={SignUpScreen}
            options={{ ...LiquidPresets.slide }}
          />
          <Stack.Screen
            name="ForgotPassword"
            component={ForgotPasswordScreen}
            options={{ ...LiquidPresets.slide }}
          />
        </>
      )}

      {/*
        ResetPassword + AuthCallback must be registered outside the auth ternary
        so they survive the isAuthenticated flip triggered by the deep-link
        session handshake. Without this, the screen unmounts before the user
        can set a new password / before the OAuth exchange finishes.
      */}
      <Stack.Screen
        name="ResetPassword"
        component={ResetPasswordScreen}
        options={{ ...LiquidPresets.slide }}
      />
      <Stack.Screen
        name="AuthCallback"
        component={AuthCallbackScreen}
        options={{ ...LiquidPresets.fade }}
      />
    </Stack.Navigator>
  );
};

export default RootNavigator;
