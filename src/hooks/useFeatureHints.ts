import { useState, useCallback, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

const HINTS_SEEN_KEY = 'aiwardrobe_hints_seen';

export interface FeatureHintData {
    id: string;
    title: string;
    description: string;
    icon: string;
    targetDescription?: string;
}

interface UseFeatureHintsReturn {
    hasSeenHint: (hintId: string) => boolean;
    markHintAsSeen: (hintId: string) => Promise<void>;
    resetAllHints: () => Promise<void>;
    seenHints: string[];
    isLoading: boolean;
}

/**
 * Hook to track iOS 26-style feature hints that users have seen
 * Persists to AsyncStorage so hints only show once per user
 */
export const useFeatureHints = (): UseFeatureHintsReturn => {
    const [seenHints, setSeenHints] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Load seen hints on mount
    useEffect(() => {
        loadSeenHints();
    }, []);

    const loadSeenHints = async () => {
        try {
            const data = await AsyncStorage.getItem(HINTS_SEEN_KEY);
            if (data) {
                setSeenHints(JSON.parse(data));
            }
        } catch (error) {
            console.error('Error loading seen hints:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const hasSeenHint = useCallback((hintId: string): boolean => {
        return seenHints.includes(hintId);
    }, [seenHints]);

    const markHintAsSeen = useCallback(async (hintId: string): Promise<void> => {
        if (seenHints.includes(hintId)) return;

        const newSeenHints = [...seenHints, hintId];
        setSeenHints(newSeenHints);

        try {
            await AsyncStorage.setItem(HINTS_SEEN_KEY, JSON.stringify(newSeenHints));
        } catch (error) {
            console.error('Error saving seen hint:', error);
        }
    }, [seenHints]);

    const resetAllHints = useCallback(async (): Promise<void> => {
        setSeenHints([]);
        try {
            await AsyncStorage.removeItem(HINTS_SEEN_KEY);
        } catch (error) {
            console.error('Error resetting hints:', error);
        }
    }, []);

    return {
        hasSeenHint,
        markHintAsSeen,
        resetAllHints,
        seenHints,
        isLoading,
    };
};

export default useFeatureHints;
