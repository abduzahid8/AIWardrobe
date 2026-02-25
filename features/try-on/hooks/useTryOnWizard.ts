/**
 * useTryOnWizard — manages wizard step state and mode toggle
 */

import { useState } from 'react';
import type { TryOnMode, TryOnStep, PhotoTab } from '../types';

export function useTryOnWizard() {
    const [tryOnMode, setTryOnMode] = useState<TryOnMode>('try your self');
    const [tryOnStep, setTryOnStep] = useState<TryOnStep>(1);
    const [activeTab, setActiveTab] = useState<PhotoTab>('upload');

    const goToStep = (step: TryOnStep) => setTryOnStep(step);
    const nextStep = () => setTryOnStep(Math.min(tryOnStep + 1, 3) as TryOnStep);
    const prevStep = () => setTryOnStep(Math.max(tryOnStep - 1, 1) as TryOnStep);

    return {
        tryOnMode,
        setTryOnMode,
        tryOnStep,
        goToStep,
        nextStep,
        prevStep,
        activeTab,
        setActiveTab,
    };
}
