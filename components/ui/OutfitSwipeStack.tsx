/**
 * A physical, tactile alternative to a horizontal outfit carousel: the current
 * day's looks sit as a small stylist's deck. Drag the top card away to decide
 * ("SAVE" right, "SKIP" left) and the next look settles forward into place.
 * The same fly-off animation is exposed imperatively so the heart / dislike
 * icon buttons can trigger the identical motion as a real swipe.
 */
import React, { forwardRef, useImperativeHandle, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { View, StyleSheet, Dimensions, Image } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  runOnJS,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { ScaledText } from './ScaledText';
import { OutfitCollagePreview, type OutfitCollageItem } from './OutfitCollagePreview';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.28;
const ROTATION_DEG = 10;
const CAPTION_BAR_HEIGHT = 52;
// Current card + the two peeking behind it, plus one more so the card that's
// about to become "back" is already warm before it's ever rendered.
const PREFETCH_LOOKAHEAD = 4;

const SPRING_BACK = { damping: 20, stiffness: 260, mass: 0.9 };
const SPRING_BACK_REDUCED = { damping: 1000, stiffness: 1000, mass: 1 };

export interface SwipeStackCard {
  key: string;
  items: OutfitCollageItem[];
  caption: string;
}

export interface OutfitSwipeStackHandle {
  swipeRight: () => void;
  swipeLeft: () => void;
}

interface OutfitSwipeStackProps {
  cards: SwipeStackCard[];
  activeIndex: number;
  onSwipeRight: (card: SwipeStackCard) => void;
  onSwipeLeft: (card: SwipeStackCard) => void;
  reducedMotion?: boolean;
  height?: number;
  emptyState?: React.ReactNode;
}

const CardFace: React.FC<{ card: SwipeStackCard }> = ({ card }) => (
  <View style={styles.cardInner}>
    <OutfitCollagePreview items={card.items} backgroundColor="#F7F9FC" footerInset={CAPTION_BAR_HEIGHT} />
    <View style={styles.captionBar}>
      <ScaledText style={styles.captionText} numberOfLines={1}>
        {card.caption}
      </ScaledText>
      <Ionicons name="bag-outline" size={15} color="rgba(10,25,49,0.4)" />
    </View>
  </View>
);

export const OutfitSwipeStack = forwardRef<OutfitSwipeStackHandle, OutfitSwipeStackProps>(
  ({ cards, activeIndex, onSwipeRight, onSwipeLeft, reducedMotion = false, height = 420, emptyState }, ref) => {
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);
    const [busy, setBusy] = useState(false);

    const topCard = cards[activeIndex] ?? null;
    const midCard = cards[activeIndex + 1] ?? null;
    const backCard = cards[activeIndex + 2] ?? null;

    // The mid/back layers only start downloading their shop-catalog photos
    // once they scroll into view, so a swipe used to reveal a card whose
    // images hadn't loaded yet — pieces would pop in one at a time over the
    // next second. Prefetching a few cards ahead keeps the deck's images
    // already warm in the native image cache by the time a swipe reveals them.
    //
    // `prefetchedRef` remembers every URI already requested. Without it,
    // this effect re-issues `Image.prefetch` for the SAME uri every time it
    // re-runs — which happens on every re-render, not just real swipes,
    // because the caller rebuilds `cards` fresh each render. Re-prefetching a
    // URI that's already showing on screen (e.g. the current top card) can
    // make that already-loaded piece flicker — visible, then blank while the
    // redundant fetch redecodes, then visible again.
    const prefetchedRef = useRef<Set<string>>(new Set());
    useEffect(() => {
      cards.slice(activeIndex, activeIndex + PREFETCH_LOOKAHEAD).forEach((card) => {
        card.items.forEach((item) => {
          if (typeof item.image !== 'string' || !item.image) return;
          const uri = item.image;
          if (prefetchedRef.current.has(uri)) return;
          prefetchedRef.current.add(uri);
          Image.prefetch(uri).catch(() => {
            prefetchedRef.current.delete(uri);
          });
        });
      });
    }, [cards, activeIndex]);

    const springConfig = reducedMotion ? SPRING_BACK_REDUCED : SPRING_BACK;
    const flyDuration = reducedMotion ? 0 : 240;

    const fireDecision = (direction: 'left' | 'right') => {
      setBusy(false);
      const card = cards[activeIndex];
      if (!card) return;
      Haptics.impactAsync(
        direction === 'right' ? Haptics.ImpactFeedbackStyle.Medium : Haptics.ImpactFeedbackStyle.Light,
      );
      if (direction === 'right') onSwipeRight(card);
      else onSwipeLeft(card);
    };

    const triggerSwipe = (direction: 'left' | 'right') => {
      if (busy || !cards[activeIndex]) return;
      setBusy(true);
      const toX = direction === 'right' ? SCREEN_WIDTH * 1.4 : -SCREEN_WIDTH * 1.4;
      // Deliberately NOT resetting translateX/Y to 0 here. That reset lands on
      // the UI thread within a frame or two, while `runOnJS(fireDecision)` has
      // to cross to the JS thread, run the parent's onSwipe* callback, update
      // `stackIndex`, and get a new render committed before `activeIndex`
      // (and therefore `topCard`) actually changes. Resetting position first
      // meant the card that was JUST swiped away — still `topCard` from
      // React's point of view for that whole gap — snapped back to full
      // visibility at center, i.e. the wrong outfit reappearing, until the
      // real next card finally landed. Position is reset in the
      // `useLayoutEffect` below instead, which only runs once `activeIndex`
      // has actually advanced, so by the time the card re-centers it's
      // already showing the correct outfit.
      translateX.value = withTiming(toX, { duration: flyDuration }, (finished) => {
        'worklet';
        if (finished) {
          runOnJS(fireDecision)(direction);
        }
      });
      translateY.value = withTiming(translateY.value * 0.4, { duration: flyDuration });
    };

    // Runs synchronously after `activeIndex` advances and the promoted card's
    // content has already committed, so the reset can never be visible on the
    // outgoing (wrong) card — see the comment in `triggerSwipe` above.
    useLayoutEffect(() => {
      translateX.value = 0;
      translateY.value = 0;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeIndex]);

    useImperativeHandle(ref, () => ({
      swipeRight: () => triggerSwipe('right'),
      swipeLeft: () => triggerSwipe('left'),
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }), [busy, activeIndex, cards]);

    const panGesture = Gesture.Pan()
      .enabled(!busy && !!topCard)
      .activeOffsetX([-10, 10])
      .failOffsetY([-14, 14])
      .onUpdate((e) => {
        translateX.value = e.translationX;
        translateY.value = e.translationY * 0.25;
      })
      .onEnd((e) => {
        const passedThreshold = Math.abs(e.translationX) > SWIPE_THRESHOLD || Math.abs(e.velocityX) > 900;
        if (passedThreshold) {
          runOnJS(triggerSwipe)(e.translationX > 0 ? 'right' : 'left');
        } else {
          translateX.value = withSpring(0, springConfig);
          translateY.value = withSpring(0, springConfig);
        }
      });

    const topCardStyle = useAnimatedStyle(() => {
      const rotate = interpolate(
        translateX.value,
        [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
        [-ROTATION_DEG, 0, ROTATION_DEG],
        Extrapolation.CLAMP,
      );
      return {
        transform: [
          { translateX: translateX.value },
          { translateY: translateY.value },
          { rotate: `${rotate}deg` },
        ],
      };
    });

    const midCardStyle = useAnimatedStyle(() => {
      const progress = interpolate(
        Math.abs(translateX.value),
        [0, SWIPE_THRESHOLD],
        [0, 1],
        Extrapolation.CLAMP,
      );
      return {
        transform: [
          { translateY: interpolate(progress, [0, 1], [14, 0]) },
          { scale: interpolate(progress, [0, 1], [0.94, 1]) },
        ],
      };
    });

    const saveStampStyle = useAnimatedStyle(() => ({
      opacity: interpolate(translateX.value, [20, SWIPE_THRESHOLD], [0, 1], Extrapolation.CLAMP),
    }));
    const skipStampStyle = useAnimatedStyle(() => ({
      opacity: interpolate(translateX.value, [-SWIPE_THRESHOLD, -20], [1, 0], Extrapolation.CLAMP),
    }));

    return (
      <View style={[styles.stage, { height }]}>
        {backCard && (
          <View style={[styles.cardSlot, styles.backSlot]}>
            <CardFace card={backCard} />
          </View>
        )}
        {midCard && (
          <Animated.View style={[styles.cardSlot, midCardStyle]}>
            <CardFace card={midCard} />
          </Animated.View>
        )}
        {topCard ? (
          <GestureDetector gesture={panGesture}>
            <Animated.View style={[styles.cardSlot, topCardStyle]}>
              <CardFace card={topCard} />
              <Animated.View pointerEvents="none" style={[styles.stamp, styles.stampSave, saveStampStyle]}>
                <Ionicons name="heart" size={15} color="#FFFFFF" />
                <ScaledText style={styles.stampSaveText}>SAVE</ScaledText>
              </Animated.View>
              <Animated.View pointerEvents="none" style={[styles.stamp, styles.stampSkip, skipStampStyle]}>
                <Ionicons name="close" size={16} color="#0A1931" />
                <ScaledText style={styles.stampSkipText}>SKIP</ScaledText>
              </Animated.View>
            </Animated.View>
          </GestureDetector>
        ) : (
          emptyState ?? null
        )}
      </View>
    );
  },
);

const styles = StyleSheet.create({
  stage: {
    width: '100%',
  },
  cardSlot: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  backSlot: {
    transform: [{ translateY: 26 }, { scale: 0.88 }],
    opacity: 0.7,
  },
  cardInner: {
    flex: 1,
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.96)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    overflow: 'hidden',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 22,
    elevation: 5,
  },
  captionBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: CAPTION_BAR_HEIGHT,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 18,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(24,58,103,0.06)',
  },
  captionText: {
    fontSize: 12,
    color: '#4D4D4D',
    flexShrink: 1,
  },
  stamp: {
    position: 'absolute',
    top: 22,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
    borderWidth: 2.5,
  },
  stampSave: {
    right: 20,
    backgroundColor: '#0A1931',
    borderColor: '#0A1931',
    transform: [{ rotate: '8deg' }],
  },
  stampSaveText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1,
  },
  stampSkip: {
    left: 20,
    backgroundColor: '#FFFFFF',
    borderColor: '#0A1931',
    transform: [{ rotate: '-8deg' }],
  },
  stampSkipText: {
    color: '#0A1931',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1,
  },
});

export default OutfitSwipeStack;
