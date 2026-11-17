import React, { useEffect, useState, useMemo, Suspense } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { ActivityIndicator, View } from "react-native";
import { scheduleScreenPreloads } from "./screenPreload";
import { scheduleIdleTask } from '../src/lib/scheduleIdleTask';
import { useNetworkRecovery } from "../hooks/useNetworkRecovery";

// Eager: only the auth shell + tab host needed for first frame.
// Everything else is lazy so the startup JS bundle stays small and
// cold start doesn't evaluate camera/video/AI/photo-picker native modules
// until the user actually navigates there (fixes 4/10 freeze on launch).
const getSignInScreen = () => require("../screens/SignInScreen").default;
const getSignUpScreen = () => require("../screens/SignUpScreen").default;
// eslint-disable-next-line @typescript-eslint/no-var-requires
const getTabNavigator = () => require("./TabNavigator").default;
const getForgotPasswordScreen = () => require("../screens/ForgotPasswordScreen").default;
const getResetPasswordScreen = () => require("../screens/ResetPasswordScreen").default;
const getAuthCallbackScreen = () => require("../screens/AuthCallbackScreen").default;
const getLanguageSelectionScreen = () => require("../screens/LanguageSelectionScreen").default;

// Share imports with idle preloading and show progress on an early first visit.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const lazyScreen = (loader: () => Promise<{ default: React.ComponentType<any> }>) => {
  let pending: ReturnType<typeof loader> | undefined;
  const preload = () => {
    if (!pending) pending = loader().catch(error => { pending = undefined; throw error; });
    return pending;
  };
  const LazyComp = React.lazy(preload);
  const Wrapped = (props: Record<string, unknown>) => (
    <Suspense fallback={<View style={{ flex: 1, backgroundColor: "#FFFFFF", alignItems: "center", justifyContent: "center" }}><ActivityIndicator accessibilityRole="progressbar" /></View>}>
      <LazyComp {...(props as object)} />
    </Suspense>
  );
  return Object.assign(Wrapped, { preload });
};

const AIOutfitmaker = lazyScreen(() => import("../screens/AIOutfitmaker"));
const WardrobeVideoScreen = lazyScreen(() => import("../screens/WardrobeVideoScreen"));
const CameraScreen = lazyScreen(() => import("../screens/CameraScreen"));
const ReviewScreen = lazyScreen(() => import("../screens/ReviewScreen"));
const OutfitCalendarScreen = lazyScreen(() => import("../screens/OutfitCalendarScreen"));
const PaywallScreen = lazyScreen(() => import("../screens/PaywallScreen"));
const OutfitInspoScreen = lazyScreen(() => import("../screens/OutfitInspoScreen"));
const CreateAvatarScreen = lazyScreen(() => import("../screens/CreateAvatarScreen"));
const MyClosetScreen = lazyScreen(() => import("../screens/MyClosetScreen"));
const PrivacyPolicyScreen = lazyScreen(() => import("../screens/PrivacyPolicyScreen"));
const TermsOfServiceScreen = lazyScreen(() => import("../screens/TermsOfServiceScreen"));
const ClothingDetailEditor = lazyScreen(() => import("../components/ClothingDetailEditor"));
const ClothingDetailScreen = lazyScreen(() => import("../screens/ClothingDetailScreen"));
const TrialExpiredScreen = lazyScreen(() => import("../screens/TrialExpiredScreen"));
const PromoCodeScreen = lazyScreen(() => import("../screens/PromoCodeScreen"));
const AdminPanelScreen = lazyScreen(() => import("../screens/AdminPanelScreen"));
const GuideScreen = lazyScreen(() => import("../screens/GuideScreen"));
const StyleQuizScreen = lazyScreen(() => import("../screens/StyleQuizScreen"));
const PurchaseIntelligenceScreen = lazyScreen(() => import("../screens/PurchaseIntelligenceScreen"));
const BodyProfileScreen = lazyScreen(() => import("../screens/BodyProfileScreen"));
import useAuthStore, { cleanupAuthSubscription, identifyRestoredPurchases } from "../store/auth";
import useSubscriptionStore, { recoverSubscriptionAfterNetworkChange } from "../store/subscriptionStore";
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
import { markAppReady } from "../src/lib/appReady";
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
  useNetworkRecovery(() => {
    const auth = useAuthStore.getState();
    if (!auth.isAuthenticated || !auth.session?.user?.id) return;
    void auth.fetchUser();
    void recoverSubscriptionAfterNetworkChange();
  });
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
      // Language and auth are independent; preserve both routing gates while
      // performing their local reads concurrently. Auth cleanup still completes
      // before any account-scoped hydration starts.
      await Promise.all([
        useLanguageStore.getState().initializeLanguage().catch((err) =>
          console.warn('[RootNavigator] initializeLanguage failed', err),
        ),
        (async () => {
          let timer: ReturnType<typeof setTimeout> | undefined;
          try {
            await Promise.race([
              useAuthStore.getState().initializeAuth(),
              new Promise<never>((_, reject) => {
                timer = setTimeout(() => reject(new Error('initializeAuth timed out after 5000ms')), 5000);
              }),
            ]);
          } catch (err) {
            console.warn('[RootNavigator] initializeAuth timeout or error — forcing isInitialized', err);
            useAuthStore.setState({ isInitialized: true, loading: false });
          } finally {
            if (timer !== undefined) clearTimeout(timer);
          }
        })(),
      ]);
      perfMeasure('app:init');

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
      const cancelIdleTasks: Array<() => void> = [];
      const backgroundTimer = setTimeout(() => {
        analyticsService.initialize();
        const { isAuthenticated: authStatus, user: currentUser } = useAuthStore.getState();
        const deferToIdle = (fn: () => void, ms = 1000) => {
          cancelIdleTasks.push(scheduleIdleTask(fn, ms));
        };
        deferToIdle(() => {
          notificationService.initialize().catch(err =>
            console.warn('[RootNavigator] notificationService failed', err));
        });
        deferToIdle(() => {
          iapService.initialize().catch(err =>
            console.warn('[RootNavigator] iapService failed', err));
        }, 1500);
        if (authStatus && currentUser?.id) {
          useWardrobeStore.getState().rehydrateFromCloud(false).catch(err => {
            if (__DEV__) console.warn('[RootNavigator] rehydrateFromCloud failed', err);
          });
        }
        // Subscription features remain pending until reconciliation completes.
        initializeSubscription().catch(err =>
          console.warn('[RootNavigator] initializeSubscription failed', err));
        if (authStatus && currentUser?.id) {
          analyticsService.setUserId(currentUser.id);
          deferToIdle(() => {
            if (useAuthStore.getState().user?.id !== currentUser.id) return;
            identifyRestoredPurchases(currentUser.id).catch(() => {});
            verifySubscriptionFromServer().catch(err =>
              console.warn('[RootNavigator] verifySubscriptionFromServer failed', err));
            hydratePromoCodeStoreSync();
          }, 2500);
        } else {
          deferToIdle(() => hydratePromoCodeStoreSync(), 2000);
        }
      }, 250);

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
        cancelIdleTasks.forEach(cancel => cancel());
        clearTimeout(backgroundTimer);
        clearTimeout(prewarmTimer);
      };
    }
  }, [isAppReady, isInitialized, isLanguageInitialized]);

  useEffect(() => {
    if (!isAppReady || !isAuthenticated || !hasCompletedOnboarding) return;
    // Share one idle queue to avoid overlapping tab and feature evaluations.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { tabScreenPreloaders } = require('./TabNavigator');
    return scheduleScreenPreloads([
      ...tabScreenPreloaders, AIOutfitmaker.preload, OutfitCalendarScreen.preload,
      ClothingDetailScreen.preload, GuideScreen.preload, PaywallScreen.preload,
      WardrobeVideoScreen.preload, CameraScreen.preload,
    ], 4000);
  }, [isAppReady, isAuthenticated, hasCompletedOnboarding]);

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
    <View style={{ flex: 1 }} onLayout={markAppReady}>
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
            <Stack.Screen name="Main" getComponent={getTabNavigator} />
            <Stack.Screen name="PurchaseIntelligence" component={PurchaseIntelligenceScreen} />

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
              component={CameraScreen}
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
            getComponent={getLanguageSelectionScreen}
            options={{ ...LiquidPresets.fade, gestureEnabled: false }}
          />
          <Stack.Screen
            name="SignIn"
            getComponent={getSignInScreen}
            // Fade visuals, but swipe-back must stay ON so the
            // language chooser underneath remains natively reachable.
            options={{ ...LiquidPresets.fade, gestureEnabled: true, gestureDirection: 'horizontal' }}
          />
          <Stack.Screen
            name="SignUp"
            getComponent={getSignUpScreen}
            options={{ ...LiquidPresets.slide }}
          />
          <Stack.Screen
            name="ForgotPassword"
            getComponent={getForgotPasswordScreen}
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
        getComponent={getResetPasswordScreen}
        options={{ ...LiquidPresets.slide }}
      />
      <Stack.Screen
        name="AuthCallback"
        getComponent={getAuthCallbackScreen}
        options={{ ...LiquidPresets.fade }}
      />
    </Stack.Navigator>
    </View>
  );
};

export default RootNavigator;
