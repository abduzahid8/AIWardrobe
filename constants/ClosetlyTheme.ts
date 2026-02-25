/**
 * constants/ClosetlyTheme.ts — Backward-compatibility re-export.
 * All tokens now live in src/theme/tokens.ts.
 */

import { StyleSheet } from 'react-native';
import { ClosetlyThemeCompat, shadows } from '../src/theme/tokens';

export const ClosetlyTheme = ClosetlyThemeCompat;

// Pre-built style objects for common patterns
export const ClosetlyStyles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: ClosetlyTheme.colors.background,
  },
  centeredContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    backgroundColor: ClosetlyTheme.colors.card,
    borderRadius: ClosetlyTheme.borderRadius.card,
    ...shadows.card,
  },
  cardSmall: {
    backgroundColor: ClosetlyTheme.colors.card,
    borderRadius: ClosetlyTheme.borderRadius.lg,
    ...shadows.cardSmall,
  },
  glassButton: {
    backgroundColor: ClosetlyTheme.colors.glassBg,
    borderRadius: ClosetlyTheme.borderRadius.button,
    paddingVertical: 16,
    paddingHorizontal: 32,
    ...shadows.button,
  },
  primaryButton: {
    backgroundColor: ClosetlyTheme.colors.text,
    borderRadius: ClosetlyTheme.borderRadius.button,
    paddingVertical: 18,
    paddingHorizontal: 40,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.button,
  },
  primaryButtonText: {
    color: ClosetlyTheme.colors.background,
    fontSize: 16,
    fontWeight: '600',
  },
  matchBadge: {
    backgroundColor: ClosetlyTheme.colors.background,
    borderRadius: ClosetlyTheme.borderRadius.sm,
    paddingVertical: 4,
    paddingHorizontal: 8,
    ...shadows.cardSmall,
  },
});

export default ClosetlyTheme;
