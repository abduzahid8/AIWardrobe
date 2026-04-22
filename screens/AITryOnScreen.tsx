/**
 * AITryOnScreen — the 3rd tab ("AI").
 *
 * Access matrix:
 *   Free → locked (FeatureLockOverlay)   — strongest upsell hook
 *   Pro  → full try-on experience from features/try-on/AITryOnScreen
 *
 * We intentionally show a tease overlay instead of redirecting/hiding
 * the tab. Users feel the feature is "one tap away" → higher conversion
 * than invisible gating.
 */

import React from 'react';
import TryOnFeatureScreen from '../features/try-on/AITryOnScreen';
import FeatureLockOverlay from '../components/paywall/FeatureLockOverlay';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';

export default function AITryOnScreen(props: any) {
    const { canAccess } = useSubscriptionGate();

    if (!canAccess('tryOns')) {
        return (
            <FeatureLockOverlay
                requiredTier="Pro"
                featureName="AI Virtual Try-On"
                tagline="See any outfit on a realistic model of yourself — before you wear it."
                icon="sparkles"
                bullets={[
                    'Unlimited photorealistic try-ons',
                    'Mix your wardrobe + shop items on one model',
                    'Share looks with friends in seconds',
                    'Priority AI model — highest quality renders',
                ]}
            />
        );
    }

    return <TryOnFeatureScreen {...props} />;
}
