/**
 * A physical, tactile alternative to a horizontal outfit carousel: the current
 * day's looks sit as a small stylist's deck. Drag the top card away to decide
 * ("SAVE" right, "SKIP" left) and the next look settles forward into place.
 * The same fly-off animation is exposed imperatively so the heart / dislike
 * icon buttons can trigger the identical motion as a real swipe.
 */
import React, { forwardRef, useImperativeHandle, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';

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
import { ExpoImage } from '../../src/components/MediaImage';


const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.28;
const ROTATION_DEG = 10;
const CAPTION_BAR_HEIGHT = 52;
// Current card + the two peeking behind it, plus one more so the card that's
// about to become "back" is already warm before it's ever rendered.
// The back card mounts immediately but was historically never prefetched
// (lookahead covered top+mid only), so every swipe revealed a mid card on
// top of a back card whose full-res photos were still downloading — the
// perpetual "wait 2-4s until everything loads". The old 4-ahead storm is
// gone now that `cards` is a stable memo (the effect below also depends on
// a content signature, not the array identity), and the dedupe set bounds
// each run to only genuinely new URIs — at most one card's worth per swipe.
const PREFETCH_LOOKAHEAD = 4;
// Bound the dedupe set so it can't grow unbounded across account switches
// (delete -> create reuses the same component instance in some navigations).
const MAX_PREFETCHED_URIS = 200;

const SPRING_BACK = { damping: 20, stiffness: 260, mass: 0.9 };
const SPRING_BACK_REDUCED = { damping: 1000, stiffness: 1000, mass: 1 };

export interface SwipeStackCard {
  key: string;
  items: OutfitCollageItem[];
  caption: string;
  wardrobeCount?: number;
  shopCount?: number;
}

export interface OutfitSwipeStackHandle {
  swipeRight: () => void;
  swipeLeft: () => void;
}

interface OutfitSwipeStackProps {
  cards: SwipeStackCard[];
  activeIndex: number;
  /**
   * Return `false` when the swipe is rejected (e.g. paywall gate) so the
   * card springs back instead of staying flown off-screen. A rejected swipe
   * without rollback desyncs the visual deck from `activeIndex` — the next
   * card shows through while Buy/share still read the old top card.
   */
  onSwipeRight: (card: SwipeStackCard) => boolean | void;
  onSwipeLeft: (card: SwipeStackCard) => boolean | void;
  reducedMotion?: boolean;
  height?: number;
  emptyState?: React.ReactNode;
}

const CardFace: React.FC<{ card: SwipeStackCard; position: number; total: number }> = ({ card, position, total }) => (
  <View style={styles.cardInner}>
    <View style={styles.cardHeader}>
      <View style={styles.wardrobePill}>
        <View style={styles.wardrobeDot} />
        <ScaledText style={styles.wardrobePillText}>
          {card.wardrobeCount ? 'YOUR WARDROBE' : 'CURATED LOOK'}
        </ScaledText>
      </View>
      <View style={styles.lookCounter} accessibilityLabel={`Look ${position} of ${total}`}>
        <ScaledText style={styles.lookCounterCurrent}>{String(position).padStart(2, '0')}</ScaledText>
        <View style={styles.lookCounterRule} />
        <ScaledText style={styles.lookCounterTotal}>{String(total).padStart(2, '0')}</ScaledText>
      </View>
    </View>
    <OutfitCollagePreview
      items={card.items}
      backgroundColor="#F2F4F0"
      footerInset={CAPTION_BAR_HEIGHT}
      variant="lookboard"
    />
    <View style={styles.captionBar}>
      <View>
        <ScaledText style={styles.captionTitle}>{card.items.length}-piece outfit</ScaledText>
        <ScaledText style={styles.captionText} numberOfLines={1}>{card.caption}</ScaledText>
      </View>
      <View style={styles.pieceCountBadge}>
        <ScaledText style={styles.pieceCountText}>{card.items.length}</ScaledText>
        <Ionicons name="shirt-outline" size={13} color="#173A65" />
      </View>
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
    // Content signature, not the array identity: the parent memoizes `cards`,
    // but any unrelated re-render with a fresh array must not re-fire network
    // work — and any real content swap must. One string compare decides.
    const cardsSignature = cards
      .map((card) => `${card.key}:${card.items.map((item) => String(item.image ?? '')).join(',')}`)
      .join('|');
    useEffect(() => {
      // Evict the dedupe set if it grew huge (e.g. across account switches)
      // so it never holds hundreds of stale URIs from a previous user.
      if (prefetchedRef.current.size > MAX_PREFETCHED_URIS) {
        prefetchedRef.current.clear();
      }
      const upcoming: string[] = [];
      // Prefetch AHEAD of the visible card only. Including `activeIndex`
      // itself re-requested URIs that <ExpoImage> was already decoding on
      // screen, which raced the on-screen decode and made the current piece
      // blink blank mid-load ("shows up, disappears, reappears"). The top
      // card's tiles are already mounted — warm only mid/back + one extra.
      cards.slice(activeIndex + 1, activeIndex + 1 + PREFETCH_LOOKAHEAD).forEach((card) => {
        card.items.forEach((item) => {
          if (typeof item.image !== 'string' || !item.image) return;
          const uri = item.image;
          if (prefetchedRef.current.has(uri)) return;
          prefetchedRef.current.add(uri);
          upcoming.push(uri);
        });
      });
      if (upcoming.length > 0) {
        // Same memory-disk store the tiles read from, so a prefetched photo
        // renders instantly on mount (and survives remounts) instead of
        // flashing blank and reloading over the network.
        ExpoImage.prefetch(upcoming, 'memory-disk').catch(() => {
          upcoming.forEach((uri) => prefetchedRef.current.delete(uri));
        });
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [cardsSignature, activeIndex]);

    const springConfig = reducedMotion ? SPRING_BACK_REDUCED : SPRING_BACK;
    const flyDuration = reducedMotion ? 0 : 240;

    const fireDecision = (direction: 'left' | 'right') => {
      setBusy(false);
      const card = cards[activeIndex];
      if (!card) return;
      Haptics.impactAsync(
        direction === 'right' ? Haptics.ImpactFeedbackStyle.Medium : Haptics.ImpactFeedbackStyle.Light,
      );
      const accepted = direction === 'right' ? onSwipeRight(card) : onSwipeLeft(card);
      if (accepted === false) {
        // Swipe was rejected (e.g. swipe quota exhausted → paywall opened).
        // The fly-off animation already completed, but `activeIndex` didn't
        // advance — spring the card back so the visual deck stays in sync
        // with the logical top card (Buy/share read from `activeIndex`).
        translateX.value = withSpring(0, springConfig);
        translateY.value = withSpring(0, springConfig);
      }
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
          <View key={`back-${backCard.key}`} style={[styles.cardSlot, styles.backSlot]}>
            <CardFace card={backCard} position={activeIndex + 3} total={cards.length} />
          </View>
        )}
        {midCard && (
          <Animated.View key={`mid-${midCard.key}`} style={[styles.cardSlot, midCardStyle]}>
            <CardFace card={midCard} position={activeIndex + 2} total={cards.length} />
          </Animated.View>
        )}
        {topCard ? (
          <GestureDetector gesture={panGesture}>
            <Animated.View key={`top-${topCard.key}`} style={[styles.cardSlot, topCardStyle]}>
              <CardFace card={topCard} position={activeIndex + 1} total={cards.length} />
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
  cardHeader: {
    position: 'absolute',
    zIndex: 10,
    top: 0,
    left: 0,
    right: 0,
    height: 48,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 14,
  },
  wardrobePill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 9,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.82)',
  },
  wardrobeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#58734E',
  },
  wardrobePillText: {
    color: '#334238',
    fontSize: 8,
    fontWeight: '800',
    letterSpacing: 0.65,
  },
  lookCounter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  lookCounterCurrent: {
    color: '#173A65',
    fontSize: 12,
    fontWeight: '800',
  },
  lookCounterRule: {
    width: 16,
    height: 1,
    backgroundColor: 'rgba(23,58,101,0.3)',
  },
  lookCounterTotal: {
    color: 'rgba(23,58,101,0.45)',
    fontSize: 10,
    fontWeight: '700',
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
    paddingHorizontal: 16,
    backgroundColor: 'rgba(255,255,255,0.96)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(24,58,103,0.06)',
  },
  captionText: {
    fontSize: 9,
    color: '#718096',
    flexShrink: 1,
  },
  captionTitle: {
    fontSize: 11,
    lineHeight: 15,
    color: '#17263B',
    fontWeight: '700',
  },
  pieceCountBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 9,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: '#EEF2F7',
  },
  pieceCountText: {
    fontSize: 10,
    color: '#173A65',
    fontWeight: '800',
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
