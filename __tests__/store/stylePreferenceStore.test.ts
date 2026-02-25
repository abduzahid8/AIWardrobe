beforeEach(() => {
  jest.clearAllMocks();
  jest.resetModules();
});

const getStore = () => {
  jest.resetModules();
  return require('../../store/stylePreferenceStore').useStylePreferenceStore;
};

describe('stylePreferenceStore', () => {
  describe('onboarding', () => {
    it('starts with hasCompletedOnboarding = false', () => {
      const useStore = getStore();
      expect(useStore.getState().hasCompletedOnboarding).toBe(false);
      expect(useStore.getState().onboardingStep).toBe(0);
    });

    it('completeOnboarding sets hasCompletedOnboarding to true', () => {
      const useStore = getStore();
      useStore.getState().completeOnboarding();
      expect(useStore.getState().hasCompletedOnboarding).toBe(true);
    });

    it('setOnboardingStep updates the step', () => {
      const useStore = getStore();
      useStore.getState().setOnboardingStep(3);
      expect(useStore.getState().onboardingStep).toBe(3);
    });

    it('resetOnboarding restores defaults', () => {
      const useStore = getStore();
      useStore.getState().completeOnboarding();
      useStore.getState().setPreferences({ favoriteColors: ['red', 'blue'] });
      useStore.getState().resetOnboarding();
      expect(useStore.getState().hasCompletedOnboarding).toBe(false);
      expect(useStore.getState().onboardingStep).toBe(0);
      expect(useStore.getState().preferences.favoriteColors).toEqual([]);
    });
  });

  describe('outfit feedback', () => {
    it('likeOutfit increments totalLikes', () => {
      const useStore = getStore();
      useStore.getState().likeOutfit('outfit-1', [], 'casual');
      expect(useStore.getState().totalLikes).toBe(1);
    });

    it('dislikeOutfit increments totalDislikes', () => {
      const useStore = getStore();
      useStore.getState().dislikeOutfit('outfit-1', [], 'casual');
      expect(useStore.getState().totalDislikes).toBe(1);
    });

    it('superLikeOutfit is counted as liked', () => {
      const useStore = getStore();
      useStore.getState().superLikeOutfit('outfit-1', [], 'casual');
      expect(useStore.getState().totalLikes).toBe(1);
    });

    it('skipOutfit does not increment likes or dislikes', () => {
      const useStore = getStore();
      useStore.getState().skipOutfit('outfit-1');
      expect(useStore.getState().totalLikes).toBe(0);
      expect(useStore.getState().totalDislikes).toBe(0);
    });

    it('trims feedback to last 500 items', () => {
      const useStore = getStore();
      for (let i = 0; i < 510; i++) {
        useStore.getState().likeOutfit(`outfit-${i}`);
      }
      expect(useStore.getState().outfitFeedback.length).toBe(500);
    });
  });

  describe('getPreferenceScore', () => {
    it('returns base score of 50 for neutral outfit', () => {
      const useStore = getStore();
      const score = useStore.getState().getPreferenceScore({ colors: [], occasion: 'unknown' });
      expect(score).toBe(50);
    });

    it('increases score for favorite color match', () => {
      const useStore = getStore();
      useStore.getState().setPreferences({ favoriteColors: ['blue'] });
      const score = useStore.getState().getPreferenceScore({ colors: ['blue'] });
      expect(score).toBeGreaterThan(50);
    });

    it('decreases score for avoided color', () => {
      const useStore = getStore();
      useStore.getState().setPreferences({ avoidColors: ['red'] });
      const score = useStore.getState().getPreferenceScore({ colors: ['red'] });
      expect(score).toBeLessThan(50);
    });

    it('clamps score between 0 and 100', () => {
      const useStore = getStore();
      useStore.getState().setPreferences({ avoidColors: ['red', 'blue', 'green', 'yellow', 'black'] });
      const score = useStore.getState().getPreferenceScore({
        colors: ['red', 'blue', 'green', 'yellow', 'black'],
      });
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });
  });

  describe('getLearnedColorPreferences', () => {
    it('returns favoriteColors as liked when no feedback', () => {
      const useStore = getStore();
      useStore.getState().setPreferences({ favoriteColors: ['navy'] });
      const { liked } = useStore.getState().getLearnedColorPreferences();
      expect(liked).toContain('navy');
    });

    it('returns avoidColors as disliked when no feedback', () => {
      const useStore = getStore();
      useStore.getState().setPreferences({ avoidColors: ['orange'] });
      const { disliked } = useStore.getState().getLearnedColorPreferences();
      expect(disliked).toContain('orange');
    });

    it('returns empty arrays when no preferences and no feedback', () => {
      const useStore = getStore();
      const { liked, disliked } = useStore.getState().getLearnedColorPreferences();
      expect(liked).toEqual([]);
      expect(disliked).toEqual([]);
    });
  });
});
