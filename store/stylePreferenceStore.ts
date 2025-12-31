import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * Style Preference Store
 * Stores user's style preferences learned from onboarding quiz and ongoing interactions
 */

export interface StylePreference {
    // Basic info
    bodyType?: 'petite' | 'tall' | 'curvy' | 'athletic' | 'average';
    stylePersonality?: 'classic' | 'trendy' | 'minimalist' | 'bohemian' | 'edgy' | 'romantic' | 'sporty';

    // Colors
    favoriteColors: string[];
    avoidColors: string[];

    // Patterns
    likedPatterns: string[];  // 'solid', 'stripes', 'floral', 'plaid', 'geometric', 'animal'
    dislikedPatterns: string[];

    // Fit preferences
    fitPreference: 'loose' | 'fitted' | 'balanced';

    // Occasions & Lifestyle
    primaryOccasions: string[];  // 'work', 'casual', 'date', 'fitness', 'formal', 'travel'
    workDressCode?: 'formal' | 'business_casual' | 'casual' | 'creative' | 'none';

    // Brands (optional)
    favoriteBrands: string[];
    avoidBrands: string[];

    // Budget
    priceRange: {
        min: number;
        max: number;
        currency: string;
    };

    // Sustainability
    prefersSustainable: boolean;

    // Goals
    styleGoals: string[];  // 'organize_closet', 'get_styled', 'shop_smarter', 'build_capsule', 'explore_trends'
}

export interface OutfitFeedback {
    outfitId: string;
    liked: boolean;
    superLiked?: boolean;
    skipped?: boolean;
    timestamp: Date;
    occasion?: string;
    items?: string[];  // Item IDs in the outfit
}

interface StylePreferenceState {
    // Preferences
    preferences: StylePreference;

    // Outfit feedback history
    outfitFeedback: OutfitFeedback[];

    // Onboarding status
    hasCompletedOnboarding: boolean;
    onboardingStep: number;

    // Derived stats
    totalLikes: number;
    totalDislikes: number;

    // Actions
    setPreferences: (prefs: Partial<StylePreference>) => void;
    addOutfitFeedback: (feedback: OutfitFeedback) => void;
    likeOutfit: (outfitId: string, items?: string[], occasion?: string) => void;
    dislikeOutfit: (outfitId: string, items?: string[], occasion?: string) => void;
    superLikeOutfit: (outfitId: string, items?: string[], occasion?: string) => void;
    skipOutfit: (outfitId: string) => void;

    // Onboarding
    completeOnboarding: () => void;
    setOnboardingStep: (step: number) => void;
    resetOnboarding: () => void;

    // Helpers
    getPreferenceScore: (outfit: any) => number;
    getLearnedColorPreferences: () => { liked: string[]; disliked: string[] };
    clearAll: () => void;
}

const DEFAULT_PREFERENCES: StylePreference = {
    favoriteColors: [],
    avoidColors: [],
    likedPatterns: [],
    dislikedPatterns: [],
    fitPreference: 'balanced',
    primaryOccasions: [],
    favoriteBrands: [],
    avoidBrands: [],
    priceRange: { min: 0, max: 500, currency: 'USD' },
    prefersSustainable: false,
    styleGoals: [],
};

export const useStylePreferenceStore = create<StylePreferenceState>()(
    persist(
        (set, get) => ({
            preferences: DEFAULT_PREFERENCES,
            outfitFeedback: [],
            hasCompletedOnboarding: false,
            onboardingStep: 0,
            totalLikes: 0,
            totalDislikes: 0,

            setPreferences: (prefs) => set((state) => ({
                preferences: { ...state.preferences, ...prefs }
            })),

            addOutfitFeedback: (feedback) => set((state) => {
                const newFeedback = [...state.outfitFeedback, feedback];
                // Keep last 500 feedback items
                const trimmedFeedback = newFeedback.slice(-500);

                const totalLikes = trimmedFeedback.filter(f => f.liked || f.superLiked).length;
                const totalDislikes = trimmedFeedback.filter(f => !f.liked && !f.skipped).length;

                return {
                    outfitFeedback: trimmedFeedback,
                    totalLikes,
                    totalDislikes
                };
            }),

            likeOutfit: (outfitId, items, occasion) => {
                get().addOutfitFeedback({
                    outfitId,
                    liked: true,
                    timestamp: new Date(),
                    items,
                    occasion
                });
            },

            dislikeOutfit: (outfitId, items, occasion) => {
                get().addOutfitFeedback({
                    outfitId,
                    liked: false,
                    timestamp: new Date(),
                    items,
                    occasion
                });
            },

            superLikeOutfit: (outfitId, items, occasion) => {
                get().addOutfitFeedback({
                    outfitId,
                    liked: true,
                    superLiked: true,
                    timestamp: new Date(),
                    items,
                    occasion
                });
            },

            skipOutfit: (outfitId) => {
                get().addOutfitFeedback({
                    outfitId,
                    liked: false,
                    skipped: true,
                    timestamp: new Date()
                });
            },

            completeOnboarding: () => set({
                hasCompletedOnboarding: true,
                onboardingStep: -1 // Completed
            }),

            setOnboardingStep: (step) => set({ onboardingStep: step }),

            resetOnboarding: () => set({
                hasCompletedOnboarding: false,
                onboardingStep: 0,
                preferences: DEFAULT_PREFERENCES
            }),

            getPreferenceScore: (outfit) => {
                const { preferences, outfitFeedback } = get();
                let score = 50; // Base score

                // Check color match
                if (outfit.colors) {
                    outfit.colors.forEach((color: string) => {
                        if (preferences.favoriteColors.includes(color.toLowerCase())) {
                            score += 10;
                        }
                        if (preferences.avoidColors.includes(color.toLowerCase())) {
                            score -= 15;
                        }
                    });
                }

                // Check occasion match
                if (outfit.occasion && preferences.primaryOccasions.includes(outfit.occasion)) {
                    score += 15;
                }

                // Check pattern match
                if (outfit.pattern) {
                    if (preferences.likedPatterns.includes(outfit.pattern)) {
                        score += 10;
                    }
                    if (preferences.dislikedPatterns.includes(outfit.pattern)) {
                        score -= 10;
                    }
                }

                // Learn from similar past feedback
                const similarFeedback = outfitFeedback.filter(f =>
                    f.occasion === outfit.occasion ||
                    f.items?.some(item => outfit.items?.includes(item))
                );

                similarFeedback.forEach(f => {
                    if (f.liked) score += 5;
                    if (f.superLiked) score += 10;
                    if (!f.liked && !f.skipped) score -= 5;
                });

                return Math.max(0, Math.min(100, score));
            },

            getLearnedColorPreferences: () => {
                const { outfitFeedback } = get();
                const colorCounts: Record<string, { liked: number; disliked: number }> = {};

                outfitFeedback.forEach(feedback => {
                    // Would need outfit details to extract colors
                    // This is a placeholder for the learning algorithm
                });

                const liked: string[] = [];
                const disliked: string[] = [];

                Object.entries(colorCounts).forEach(([color, counts]) => {
                    if (counts.liked > counts.disliked * 2) {
                        liked.push(color);
                    } else if (counts.disliked > counts.liked * 2) {
                        disliked.push(color);
                    }
                });

                return { liked, disliked };
            },

            clearAll: () => set({
                preferences: DEFAULT_PREFERENCES,
                outfitFeedback: [],
                hasCompletedOnboarding: false,
                onboardingStep: 0,
                totalLikes: 0,
                totalDislikes: 0
            })
        }),
        {
            name: 'style-preferences',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);

export default useStylePreferenceStore;
