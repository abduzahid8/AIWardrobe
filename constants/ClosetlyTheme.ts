/**
 * Closetly Theme Constants
 * MONOCHROME - Pure Black and White Only
 */

import { StyleSheet, Platform } from 'react-native';

export const ClosetlyTheme = {
  // Color Palette - BLACK AND WHITE ONLY
  colors: {
    background: '#FFFFFF',
    card: '#F5F5F5',
    cardHover: '#EEEEEE',
    text: '#000000',
    textSecondary: '#4D4D4D',
    textMuted: '#808080',
    overlay: 'rgba(255, 255, 255, 0.95)',
    glassBg: 'rgba(255, 255, 255, 0.85)',
    success: '#000000',
    matchHighlight: '#000000',
  },

  // Spacing system
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
    cardGap: 16,
    screenPadding: 20,
    sectionGap: 24,  // White space between split sections
  },

  // Exaggerated rounded corners for premium feel
  borderRadius: {
    sm: 12,
    md: 20,
    lg: 24,
    xl: 32,
    card: 32,
    button: 24,
    pill: 50,
  },

  // Soft, diffused shadows for floating effect
  shadows: {
    card: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.08,
      shadowRadius: 24,
      elevation: 8,
    },
    cardSmall: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.06,
      shadowRadius: 12,
      elevation: 4,
    },
    button: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.1,
      shadowRadius: 16,
      elevation: 6,
    },
    floating: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 12 },
      shadowOpacity: 0.12,
      shadowRadius: 32,
      elevation: 12,
    },
  },

  // Typography - SF Pro Display style
  typography: {
    // Massive emotional headers (32px+)
    hero: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 36,
      fontWeight: '700' as const,
      letterSpacing: -0.5,
      color: '#000000',
    },
    header: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 32,
      fontWeight: '700' as const,
      letterSpacing: -0.3,
      color: '#000000',
    },
    title: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 24,
      fontWeight: '600' as const,
      letterSpacing: -0.2,
      color: '#000000',
    },
    subtitle: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 18,
      fontWeight: '600' as const,
      color: '#000000',
    },
    // Small utilitarian labels
    body: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 16,
      fontWeight: '400' as const,
      color: '#666666',
    },
    label: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 12,
      fontWeight: '500' as const,
      letterSpacing: 0.5,
      textTransform: 'uppercase' as const,
      color: '#999999',
    },
    caption: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 11,
      fontWeight: '500' as const,
      color: '#999999',
    },
    // Match percentage badge
    matchScore: {
      fontFamily: Platform.OS === 'ios' ? 'System' : 'sans-serif',
      fontSize: 14,
      fontWeight: '700' as const,
      color: '#000000',
    },
  },

  // Animation durations
  animation: {
    fast: 150,
    normal: 250,
    slow: 400,
    spring: {
      damping: 15,
      stiffness: 150,
      mass: 1,
    },
  },

  // Card dimensions for clothing items
  cardDimensions: {
    carouselItem: {
      width: 140,
      height: 180,
    },
    carouselItemLarge: {
      width: 160,
      height: 200,
    },
    modelView: {
      width: 200,
      height: 280,
    },
  },
};

// Pre-built style objects for common patterns
export const ClosetlyStyles = StyleSheet.create({
  // Container styles
  screen: {
    flex: 1,
    backgroundColor: ClosetlyTheme.colors.background,
  },

  centeredContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  // Card styles
  card: {
    backgroundColor: ClosetlyTheme.colors.card,
    borderRadius: ClosetlyTheme.borderRadius.card,
    ...ClosetlyTheme.shadows.card,
  },

  cardSmall: {
    backgroundColor: ClosetlyTheme.colors.card,
    borderRadius: ClosetlyTheme.borderRadius.lg,
    ...ClosetlyTheme.shadows.cardSmall,
  },

  // Glassmorphism button
  glassButton: {
    backgroundColor: ClosetlyTheme.colors.glassBg,
    borderRadius: ClosetlyTheme.borderRadius.button,
    paddingVertical: 16,
    paddingHorizontal: 32,
    ...ClosetlyTheme.shadows.button,
  },

  // Primary action button
  primaryButton: {
    backgroundColor: ClosetlyTheme.colors.text,
    borderRadius: ClosetlyTheme.borderRadius.button,
    paddingVertical: 18,
    paddingHorizontal: 40,
    alignItems: 'center',
    justifyContent: 'center',
    ...ClosetlyTheme.shadows.button,
  },

  primaryButtonText: {
    color: ClosetlyTheme.colors.background,
    fontSize: 16,
    fontWeight: '600',
  },

  // Match score badge
  matchBadge: {
    backgroundColor: ClosetlyTheme.colors.background,
    borderRadius: ClosetlyTheme.borderRadius.sm,
    paddingVertical: 4,
    paddingHorizontal: 8,
    ...ClosetlyTheme.shadows.cardSmall,
  },
});

export default ClosetlyTheme;
