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
import { useTranslation } from 'react-i18next';

export default function AITryOnScreen(props: any) {
    const { t } = useTranslation();
    const { canAccess } = useSubscriptionGate();

    if (!canAccess('tryOns')) {
        return (
            <FeatureLockOverlay
                requiredTier="Pro"
                featureName={t('aiTryOn.featureName')}
                tagline={t('aiTryOn.tagline')}
                icon="sparkles"
                bullets={[
                    t('aiTryOn.bullet1'),
                    t('aiTryOn.bullet2'),
                    t('aiTryOn.bullet3'),
                    t('aiTryOn.bullet4'),
                ]}
            />
        );
    }

    return <TryOnFeatureScreen {...props} />;
}
