import React, { useEffect, useRef, useState, useMemo } from 'react';
import { View, Animated, Dimensions, StyleSheet, Easing } from 'react-native'

import { ScaledText } from './ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useTranslation } from 'react-i18next';
import { ExpoImage as Image } from '../src/components/MediaImage';


const { width, height } = Dimensions.get('window');

interface ClothingItem {
  id: string;
  image: string;
  name?: string;
  type?: string;
  category?: string;
}

interface ThinkingStep {
  id: number;
  text: string;
  icon: any;
}

interface AIThinkingAnimationProps {
  styleName?: string;
  clothingItems?: ClothingItem[];
}

interface OrbitItem {
  id: number;
  imageIndex: number;
  angleDeg: number;
  size: number;
  opacity: number;
}

// Build N evenly-spaced items for one ring. Deterministic sizes so
// memoization holds across renders.
const buildRingItems = (
  count: number,
  startId: number,
  imageCount: number,
  offsetDeg: number,
): OrbitItem[] => {
  const items: OrbitItem[] = [];
  for (let k = 0; k < count; k++) {
    const id = startId + k;
    items.push({
      id,
      imageIndex: imageCount > 0 ? id % imageCount : 0,
      angleDeg: (k / count) * 360 + offsetDeg,
      size: 52 + (id % 4) * 7,
      opacity: 0.9 + ((id % 3) * 0.05),
    });
  }
  return items;
};

// One clothing thumbnail, absolutely positioned inside its ring square.
// The parent ring rotates (orbit); this cell counter-rotates with the same
// shared animated node so the garment stays upright while revolving.
const OrbitThumb = React.memo(function OrbitThumb({
  left,
  top,
  size,
  opacity,
  counterSpin,
  uri,
}: {
  left: number;
  top: number;
  size: number;
  opacity: number;
  counterSpin: Animated.AnimatedInterpolation<string>;
  uri: string;
}) {
  const h = size * 1.2;
  return (
    <Animated.View
      style={{
        position: 'absolute',
        left,
        top,
        width: size,
        height: h,
        transform: [{ rotate: counterSpin }],
      }}
    >
      <View style={[styles.helixItem, { width: size, height: h, opacity }]}>
        <Image
          source={{ uri }}
          style={styles.clothingImage}
          contentFit="cover"
          cachePolicy="memory-disk"
          recyclingKey={`${uri}-${left}-${top}`}
        />
      </View>
    </Animated.View>
  );
});

export const AIThinkingAnimation: React.FC<AIThinkingAnimationProps> = ({
  styleName,
  clothingItems = [],
}) => {
  const { t } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  // Real container size — makes the animation truly full-screen inside
  // the flex area (not window size, which includes header + safe areas).
  const [layout, setLayout] = useState({ w: width, h: height });
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const stepFadeAnim = useRef(new Animated.Value(1)).current;
  const floatAnim = useRef(new Animated.Value(0));
  // One looping timer per orbit ring — shared by all items on that ring.
  const orbitOuterAnim = useRef(new Animated.Value(0));
  const orbitInnerAnim = useRef(new Animated.Value(0));

  const THINKING_STEPS: ThinkingStep[] = [
    { id: 1, text: t('aiThinking.scanningWardrobe'), icon: 'scan-outline' },
    { id: 2, text: t('aiThinking.analyzingStyle'), icon: 'color-palette-outline' },
    { id: 3, text: t('aiThinking.matchingCombinations'), icon: 'git-merge-outline' },
    { id: 4, text: t('aiThinking.optimizingOutfits'), icon: 'options-outline' },
    { id: 5, text: t('aiThinking.finalizingLooks'), icon: 'sparkles' },
  ];

  // Default fallback images
  const clothingImages = useMemo(() => {
    if (clothingItems.length > 0) return clothingItems.map(item => item.image).filter(Boolean);
    return [
      'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=200',
      'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=200',
      'https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=200',
      'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=200',
      'https://images.unsplash.com/photo-1549298916-b41fb501e0de?w=200',
      'https://images.unsplash.com/photo-1603252109303-275144992cd7?w=200',
      'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200',
      'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=200',
    ];
  }, [clothingItems]);

  // 12 garments total: 6 outer + 6 inner. All 12 are always rendered —
  // images cycle when the wardrobe has fewer photos.
  const outerItems = useMemo(
    () => buildRingItems(6, 0, clothingImages.length, 0),
    [clothingImages.length],
  );
  const innerItems = useMemo(
    () => buildRingItems(6, 100, clothingImages.length, 30),
    [clothingImages.length],
  );

  // Orbit geometry — centered exactly on the СТИЛЬ DNA circle.
  // The brain is the only flex child (everything else is absolute), so it
  // sits at the container center (w/2, h/2). Rings use the same point as
  // their rotation pivot, so garments revolve around the circle itself.
  const centerX = layout.w / 2;
  const centerY = layout.h / 2;
  const outerRadius = Math.min(layout.w * 0.36, layout.h * 0.22);
  const innerRadius = outerRadius * 0.62;

  // Position each item on its ring circle (static layout; rotation animates).
  const outerLayout = useMemo(
    () =>
      outerItems.map(item => {
        const rad = (item.angleDeg * Math.PI) / 180;
        return {
          ...item,
          left: outerRadius + Math.cos(rad) * outerRadius - item.size / 2,
          top: outerRadius + Math.sin(rad) * outerRadius - (item.size * 1.2) / 2,
          uri: clothingImages[item.imageIndex % clothingImages.length] ?? '',
        };
      }),
    [outerItems, clothingImages, outerRadius],
  );
  const innerLayout = useMemo(
    () =>
      innerItems.map(item => {
        const rad = (item.angleDeg * Math.PI) / 180;
        return {
          ...item,
          left: innerRadius + Math.cos(rad) * innerRadius - item.size / 2,
          top: innerRadius + Math.sin(rad) * innerRadius - (item.size * 1.2) / 2,
          uri: clothingImages[item.imageIndex % clothingImages.length] ?? '',
        };
      }),
    [innerItems, clothingImages, innerRadius],
  );

  // Shared native-driven rotations. Outer spins clockwise, inner
  // counter-clockwise. 0deg == 360deg so the loop restart is seamless.
  const outerSpin = useMemo(
    () => orbitOuterAnim.current.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const outerCounter = useMemo(
    () => orbitOuterAnim.current.interpolate({ inputRange: [0, 1], outputRange: ['360deg', '0deg'] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const innerSpin = useMemo(
    () => orbitInnerAnim.current.interpolate({ inputRange: [0, 1], outputRange: ['360deg', '0deg'] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const innerCounter = useMemo(
    () => orbitInnerAnim.current.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    // Native-driven loops only (float + pulse + 2 orbit rings) — no
    // per-item JS timers, no per-frame setState.
    const floatLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(floatAnim.current, {
          toValue: 1,
          duration: 3000,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
        Animated.timing(floatAnim.current, {
          toValue: 0,
          duration: 3000,
          easing: Easing.inOut(Easing.sin),
          useNativeDriver: true,
        }),
      ])
    );
    floatLoop.start();

    const pulseLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    pulseLoop.start();

    // Clothing orbit — linear 0→1 loops, seamless (0deg == 360deg).
    // Outer clockwise (24s), inner counter-clockwise (17s) for DNA feel.
    const outerLoop = Animated.loop(
      Animated.timing(orbitOuterAnim.current, {
        toValue: 1,
        duration: 24000,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    );
    outerLoop.start();

    const innerLoop = Animated.loop(
      Animated.timing(orbitInnerAnim.current, {
        toValue: 1,
        duration: 17000,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    );
    innerLoop.start();

    // Steps — state only, no nested Animated.sequence inside setState.
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= THINKING_STEPS.length - 1) return prev;
        stepFadeAnim.setValue(0);
        Animated.timing(stepFadeAnim, { toValue: 1, duration: 200, useNativeDriver: true }).start();
        return prev + 1;
      });
    }, 1800);

    return () => {
      clearInterval(stepInterval);
      floatLoop.stop();
      pulseLoop.stop();
      outerLoop.stop();
      innerLoop.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const floatY = floatAnim.current.interpolate({ inputRange: [0, 1], outputRange: [0, -12] });

  const currentStepData = THINKING_STEPS[currentStep];
  const progressPct = Math.round((currentStep / (THINKING_STEPS.length - 1)) * 100);

  const onLayout = useRef(({ nativeEvent }: any) => {
    const { width: w, height: h } = nativeEvent.layout;
    setLayout(prev => (Math.abs(w - prev.w) > 1 || Math.abs(h - prev.h) > 1 ? { w, h } : prev));
  }).current;

  return (
    <View style={styles.container} onLayout={onLayout}>
      <LinearGradient colors={['rgba(43, 92, 233, 0.08)', 'rgba(236, 72, 153, 0.05)', 'transparent']} style={StyleSheet.absoluteFill} />

      {/* Orbit layer — 12 garments on 2 rings spinning around the brain */}
      <Animated.View style={[styles.orbitLayer, { transform: [{ translateY: floatY }] }]} pointerEvents="none">
        {/* Outer ring — square centered on the brain, rotates as one unit */}
        <Animated.View
          style={{
            position: 'absolute',
            left: centerX - outerRadius,
            top: centerY - outerRadius,
            width: outerRadius * 2,
            height: outerRadius * 2,
            transform: [{ rotate: outerSpin }],
          }}
        >
          {outerLayout.map(item => (
            <OrbitThumb
              key={`o-${item.id}`}
              left={item.left}
              top={item.top}
              size={item.size}
              opacity={item.opacity}
              counterSpin={outerCounter}
              uri={item.uri}
            />
          ))}
        </Animated.View>

        {/* Inner ring — opposite direction */}
        <Animated.View
          style={{
            position: 'absolute',
            left: centerX - innerRadius,
            top: centerY - innerRadius,
            width: innerRadius * 2,
            height: innerRadius * 2,
            transform: [{ rotate: innerSpin }],
          }}
        >
          {innerLayout.map(item => (
            <OrbitThumb
              key={`i-${item.id}`}
              left={item.left}
              top={item.top}
              size={item.size}
              opacity={item.opacity}
              counterSpin={innerCounter}
              uri={item.uri}
            />
          ))}
        </Animated.View>
      </Animated.View>

      {/* Center AI Core */}
      <Animated.View style={[styles.centerBrain, { transform: [{ scale: pulseAnim }] }]}>
        <View style={styles.brainInner}>
          <View style={styles.aiIconContainer}>
            <Ionicons name="flask" size={32} color="#2B5CE9" />
          </View>
          <ScaledText style={styles.brainText}>{t('aiThinking.styleDna')}</ScaledText>
        </View>
      </Animated.View>

      {/* Thinking text */}
      <View style={styles.textContainer}>
        <Animated.View style={{ opacity: stepFadeAnim }}>
          <View style={styles.stepRow}>
            <Ionicons name={currentStepData.icon} size={20} color="#2B5CE9" style={styles.stepIcon} />
            <ScaledText style={styles.stepText}>{currentStepData.text}</ScaledText>
          </View>
        </Animated.View>
        {styleName && <ScaledText style={styles.styleTag}>{t('aiThinking.synthesizingDna', { style: styleName })}</ScaledText>}
      </View>

      {/* Progress bar — static step width (5 updates total) */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBackground}>
          <View style={[styles.progressFill, { width: `${progressPct}%` }]} />
        </View>
        <ScaledText style={styles.progressText}>{progressPct}%</ScaledText>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  orbitLayer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    overflow: 'hidden',
  },
  helixItem: {
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#fff',
    elevation: 2,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.9)',
  },
  clothingImage: {
    width: '100%',
    height: '100%',
  },
  centerBrain: {
    width: 124,
    height: 124,
    borderRadius: 62,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: 'rgba(43, 92, 233, 0.4)',
    overflow: 'hidden',
    elevation: 6,
    zIndex: 20,
    backgroundColor: '#FFFFFF',
  },
  brainInner: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(43, 92, 233, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 6,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.2)',
  },
  brainText: {
    fontSize: 10,
    fontWeight: '900',
    color: '#2B5CE9',
    letterSpacing: 1.2,
  },
  textContainer: {
    position: 'absolute',
    bottom: 138,
    left: 20,
    right: 20,
    alignItems: 'center',
    zIndex: 10,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    maxWidth: width - 40,
    backgroundColor: 'rgba(255,255,255,0.95)',
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.25)',
    elevation: 3,
  },
  stepIcon: {
    marginRight: 8,
  },
  stepText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1F2937',
    flexShrink: 1,
  },
  styleTag: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 16,
    fontWeight: '500',
  },
  progressContainer: {
    position: 'absolute',
    bottom: 24,
    left: 40,
    right: 40,
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 10,
  },
  progressBackground: {
    flex: 1,
    height: 8,
    backgroundColor: 'rgba(229, 231, 235, 0.8)',
    borderRadius: 4,
    overflow: 'hidden',
    marginRight: 12,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#2B5CE9',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2B5CE9',
    minWidth: 40,
  },
});

export default AIThinkingAnimation;
