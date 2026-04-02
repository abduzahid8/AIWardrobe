/**
 * useTryOnWizard — manages wizard step state and mode toggle
 */

import { useState } from 'react';
import type { TryOnMode, TryOnStep, PhotoTab } from '../types';

export function useTryOnWizard() {
    const [tryOnMode, setTryOnMode] = useState<TryOnMode>('try your self');
    const [tryOnStep, setTryOnStep] = useState<TryOnStep>(1);
    const [activeTab, setActiveTab] = useState<PhotoTab>('shop');

    const goToStep = (step: TryOnStep) => setTryOnStep(step);

    return {
        tryOnMode,
        setTryOnMode,
        tryOnStep,
        goToStep,
        activeTab,
        setActiveTab,
    };
}
