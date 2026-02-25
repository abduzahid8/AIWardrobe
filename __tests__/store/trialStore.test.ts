import AsyncStorage from '@react-native-async-storage/async-storage';

// Reset modules before each test to get a fresh store
beforeEach(() => {
  jest.resetAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

// Re-import store fresh each test via jest.isolateModules
const getStore = () => {
  return require('../../store/trialStore').default;
};

describe('trialStore', () => {
  it('initializes with zero trial count when AsyncStorage is empty', async () => {
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    expect(useTrialStore.getState().trialCount).toBe(0);
    expect(useTrialStore.getState().isTrialExpired).toBe(false);
  });

  it('reads stored trial count from AsyncStorage on init', async () => {
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'trial_count') return Promise.resolve('3');
      if (key === 'trial_first_launch') return Promise.resolve(new Date().toISOString());
      return Promise.resolve(null);
    });
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    expect(useTrialStore.getState().trialCount).toBe(3);
    expect(useTrialStore.getState().isTrialExpired).toBe(false);
  });

  it('marks trial as expired when count reaches MAX (5)', async () => {
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'trial_count') return Promise.resolve('5');
      return Promise.resolve(null);
    });
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    expect(useTrialStore.getState().isTrialExpired).toBe(true);
  });

  it('increments trial count and persists to AsyncStorage', async () => {
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    await useTrialStore.getState().incrementTrialCount();
    expect(useTrialStore.getState().trialCount).toBe(1);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('trial_count', '1');
  });

  it('getTrialsRemaining returns correct remaining count', async () => {
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'trial_count') return Promise.resolve('2');
      return Promise.resolve(null);
    });
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    expect(useTrialStore.getState().getTrialsRemaining()).toBe(3);
  });

  it('getTrialsRemaining returns 0 when expired', async () => {
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'trial_count') return Promise.resolve('5');
      return Promise.resolve(null);
    });
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    expect(useTrialStore.getState().getTrialsRemaining()).toBe(0);
  });

  it('resetTrial clears AsyncStorage and resets state', async () => {
    const useTrialStore = getStore();
    await useTrialStore.getState().initializeTrial();
    await useTrialStore.getState().incrementTrialCount();
    await useTrialStore.getState().resetTrial();
    expect(useTrialStore.getState().trialCount).toBe(0);
    expect(useTrialStore.getState().isTrialExpired).toBe(false);
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('trial_count');
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('trial_first_launch');
  });
});
