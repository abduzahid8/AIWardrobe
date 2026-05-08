// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
);

// Mock expo-haptics
jest.mock('expo-haptics', () => ({
  impactAsync: jest.fn(),
  notificationAsync: jest.fn(),
  ImpactFeedbackStyle: { Light: 'light', Medium: 'medium', Heavy: 'heavy' },
  NotificationFeedbackType: { Success: 'success', Warning: 'warning', Error: 'error' },
}));

// Mock expo-location
jest.mock('expo-location', () => ({
  requestForegroundPermissionsAsync: jest.fn().mockResolvedValue({ status: 'granted' }),
  getCurrentPositionAsync: jest.fn().mockResolvedValue({
    coords: { latitude: 41.2995, longitude: 69.2401 },
  }),
}));

// Mock supabase
jest.mock('./lib/supabase', () => ({
  supabase: {
    auth: {
      getSession: jest.fn().mockResolvedValue({ data: { session: null }, error: null }),
      signUp: jest.fn(),
      signInWithPassword: jest.fn(),
      signOut: jest.fn(),
      onAuthStateChange: jest.fn().mockReturnValue({ data: { subscription: { unsubscribe: jest.fn() } } }),
    },
    from: jest.fn().mockReturnValue({
      select: jest.fn().mockReturnThis(),
      eq: jest.fn().mockReturnThis(),
      in: jest.fn().mockReturnThis(),
      single: jest.fn().mockResolvedValue({ data: null, error: null }),
    }),
  },
}));

// Mock src/config/env
jest.mock('./src/config/env', () => ({
  Config: {
    supabase: { url: 'https://test.supabase.co', anonKey: 'test-key' },
    api: { url: 'http://localhost:3000', alicevisionUrl: 'http://localhost:5050' },
    revenueCat: { apiKey: 'test-revenuecat-key' },
    weather: { apiKey: 'test-weather-key', baseUrl: 'https://api.openweathermap.org/data/2.5' },
    sentry: { dsn: '' },
    admin: { email: 'info@aiwardrobe.club' },
  },
  default: {
    supabase: { url: 'https://test.supabase.co', anonKey: 'test-key' },
    api: { url: 'http://localhost:3000', alicevisionUrl: 'http://localhost:5050' },
    revenueCat: { apiKey: 'test-revenuecat-key' },
    weather: { apiKey: 'test-weather-key', baseUrl: 'https://api.openweathermap.org/data/2.5' },
    sentry: { dsn: '' },
    admin: { email: 'info@aiwardrobe.club' },
  },
}));
