import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Animated, AppState, type AppStateStatus, Easing, StyleSheet, useWindowDimensions, View } from 'react-native';
import * as SplashScreen from 'expo-splash-screen';
import { onAppReady } from '../lib/appReady';
import { Image } from './MediaImage';


// Brand splash look — must stay visually identical to the native splash
// (iOS SplashScreen.storyboard + Android splash_background.xml) so the
// handoff between native splash and this overlay is seamless.
const BRAND_BACKGROUND = '#ADD8E6';
const LOGO_BOX_RATIO = 0.56; // logo box width as a fraction of screen width
const HIDE_DELAY_MS = 100; // beat after the first frame so the handoff is seamless
const OPEN_FADE_MS = 420;
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
  const appState = useRef<AppStateStatus>(AppState.currentState);
  const handOffStarted = useRef(false);
  const fadedInForBackground = useRef(false);

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
        if (finished) setIsVisible(false);
      });
    },
    [opacity, scale]
  );

  // App open: once RootNavigator signals readiness, keep the branded
  // overlay up for a beat (covering the first real frame), hide the native
  // splash underneath it, then dissolve into the app.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = onAppReady(() => {
      if (handOffStarted.current) return;
      handOffStarted.current = true;
      timer = setTimeout(() => {
        SplashScreen.hideAsync().catch(() => undefined);
        fadeOut(OPEN_FADE_MS);
      }, HIDE_DELAY_MS);
    });
    return () => {
      unsubscribe();
      if (timer) clearTimeout(timer);
    };
  }, [fadeOut]);

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
        fadeOut(OPEN_FADE_MS);
      }
    });
    return () => subscription.remove();
  }, [fadeOut, opacity, scale]);

  if (!isVisible) return null;

  const boxSize = Math.round(width * LOGO_BOX_RATIO);

  return (
    <Animated.View
      pointerEvents="none"
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={[
        styles.overlay,
        { backgroundColor: BRAND_BACKGROUND, opacity, transform: [{ scale }] },
      ]}
    >
      <View style={[styles.logoBox, { width: boxSize, height: boxSize }]}>
        <Image
          source={require('../../assets/splash-logo.png')}
          style={styles.logo}
          resizeMode="contain"
          accessible={false}
        />
      </View>
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
