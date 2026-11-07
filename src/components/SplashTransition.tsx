import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, Animated, AppState, type AppStateStatus, Easing, StyleSheet, useWindowDimensions } from 'react-native';
import * as SplashScreen from 'expo-splash-screen';
import { onAppReady } from '../lib/appReady';
import { recordStartupVisible } from '../lib/startupTiming';
import { Image } from 'react-native';


// Brand splash look — must stay visually identical to the native splash
// (iOS SplashScreen.storyboard + Android splash_background.xml) so the
// handoff between native splash and this overlay is seamless.
const BRAND_BACKGROUND = '#C4DCFB';
const LOGO_BOX_RATIO = 0.56; // logo box width as a fraction of screen width
const OPEN_FADE_MS = 160;
const CLOSE_FADE_MS = 240;
const EXIT_SCALE = 1.04; // barely-there "breath" as the app reveals itself

/**
 * Animated stand-in for the native splash screen.
 *
 * - App open: mounted just before the native splash is hidden, then fades
 *   out (with a subtle scale) to reveal the first real screen.
 * - App close/background: fades the branded splash back in, so the app
 *   switcher snapshot and the return-to-foreground transition feel polished.
 *
 * The overlay never intercepts touches (pointerEvents="none") and unmounts
 * itself once invisible.
 */
const SplashTransition: React.FC = () => {
  const { width } = useWindowDimensions();
  const opacity = useRef(new Animated.Value(1)).current;
  const scale = useRef(new Animated.Value(1)).current;
  const [isVisible, setIsVisible] = useState(true);
  const [logoLoaded, setLogoLoaded] = useState(false);
  const [overlayLaidOut, setOverlayLaidOut] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(true);
  const logoPulse = useRef(new Animated.Value(0)).current;
  const appState = useRef<AppStateStatus>(AppState.currentState);
  const handOffStarted = useRef(false);
  const fadedInForBackground = useRef(false);
  const startupRecorded = useRef(false);

  const fadeOut = useCallback(
    (duration: number) => {
      Animated.parallel([
        Animated.timing(opacity, {
          toValue: 0,
          duration,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(scale, {
          toValue: EXIT_SCALE,
          duration,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start(({ finished }) => {
        if (finished) {
          setIsVisible(false);
          if (!startupRecorded.current) {
            startupRecorded.current = true;
            recordStartupVisible();
          }
        }
      });
    },
    [opacity, scale]
  );

  useEffect(() => {
    let active = true;
    const subscription = AccessibilityInfo.addEventListener('reduceMotionChanged', setReduceMotion);
    AccessibilityInfo.isReduceMotionEnabled()
      .then((enabled) => { if (active) setReduceMotion(enabled); })
      .catch(() => undefined);
    return () => { active = false; subscription.remove(); };
  }, []);

  useEffect(() => {
    if (!isVisible || reduceMotion) {
      logoPulse.setValue(0);
      return;
    }
    const animation = Animated.loop(Animated.sequence([
      Animated.timing(logoPulse, {
        toValue: 1, duration: 700, easing: Easing.inOut(Easing.sin), useNativeDriver: true,
      }),
      Animated.timing(logoPulse, {
        toValue: 0, duration: 700, easing: Easing.inOut(Easing.sin), useNativeDriver: true,
      }),
    ]));
    animation.start();
    return () => { animation.stop(); logoPulse.setValue(0); };
  }, [isVisible, logoPulse, reduceMotion]);

  // Reveal the animated overlay as soon as its logo is painted. Keep it
  // visible while initialization runs, without adding an artificial delay.
  useEffect(() => {
    if (!logoLoaded || !overlayLaidOut) return;
    let cancelled = false;
    let unsubscribe: (() => void) | undefined;
    SplashScreen.hideAsync().catch(() => undefined).then(() => {
      if (cancelled) return;
      unsubscribe = onAppReady(() => {
        if (handOffStarted.current) return;
        handOffStarted.current = true;
        fadeOut(reduceMotion ? 0 : OPEN_FADE_MS);
      });
    });
    return () => { cancelled = true; unsubscribe?.(); };
  }, [fadeOut, logoLoaded, overlayLaidOut, reduceMotion]);

  // App close / reopen: subtle branded fade on backgrounding, dissolve
  // again when the app returns to the foreground. iOS walks
  // active → inactive → background (and back), so compare against the
  // state we came from rather than requiring an exact 'active' neighbor.
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (next) => {
      const previous = appState.current;
      appState.current = next;

      if (next === 'background' && previous !== 'background') {
        opacity.stopAnimation();
        scale.stopAnimation();
        opacity.setValue(0);
        scale.setValue(1);
        fadedInForBackground.current = true;
        setIsVisible(true);
        Animated.timing(opacity, {
          toValue: 1,
          duration: CLOSE_FADE_MS,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        }).start();
      } else if (next === 'active' && fadedInForBackground.current) {
        fadedInForBackground.current = false;
        fadeOut(reduceMotion ? 0 : OPEN_FADE_MS);
      }
    });
    return () => subscription.remove();
  }, [fadeOut, opacity, scale, reduceMotion]);

  if (!isVisible) return null;

  const boxSize = Math.round(width * LOGO_BOX_RATIO);

  return (
    <Animated.View
      onLayout={() => setOverlayLaidOut(true)}
      pointerEvents="none"
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={[
        styles.overlay,
        { backgroundColor: BRAND_BACKGROUND, opacity, transform: [{ scale }] },
      ]}
    >
      <Animated.View style={[styles.logoBox, {
        width: boxSize, height: boxSize,
        transform: [{ scale: logoPulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.06] }) }],
      }]}>
        <Image
          source={require('../../assets/splash-logo.png')}
          style={styles.logo}
          onLoad={() => setLogoLoaded(true)}
          onError={() => setLogoLoaded(true)}
          fadeDuration={0}
          resizeMode="contain"
          accessible={false}
        />
      </Animated.View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 999,
    elevation: 999,
  },
  logoBox: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: '100%',
    height: '100%',
  },
});

export default SplashTransition;
