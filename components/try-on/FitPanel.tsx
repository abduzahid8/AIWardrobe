/**
 * FitPanel — renders a FitAssessment as a glanceable card.
 *
 * Sections:
 *   1. Overall badge (good_fit / tight / relaxed / oversized / too_small / too_large)
 *      with a confidence chip (low / medium / high)
 *   2. Optional size recommendation ("We recommend size M")
 *   3. Per-zone notes — green ✓ for good, amber ⚠ for snug/loose, red ✕ for too tight/loose
 *   4. Empty / loading / error states
 *
 * Designed to drop into any try-on flow. Stateless — pass the assessment
 * and it renders. No fetching, no state.
 *
 * Visual style mirrors the rest of the app (system fonts, light grey cards).
 */

import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useTranslation } from 'react-i18next';

import type {
  FitAssessment,
  FitZoneAssessment,
  OverallFit,
  Confidence,
  ZoneStatus,
} from '../../src/types/fitAssessment';

export interface FitPanelProps {
  assessment?: FitAssessment;
  loading?: boolean;
  error?: string | null;
  /** When the engine has no data at all, show a CTA to set up the body profile. */
  onSetupBodyProfile?: () => void;
}

const FitPanel: React.FC<FitPanelProps> = ({
  assessment, loading, error, onSetupBodyProfile,
}) => {
  const { t } = useTranslation();

  if (loading) {
    return (
      <View style={styles.card}>
        <View style={styles.headerRow}>
          <ActivityIndicator size="small" />
          <Text style={styles.loadingText}>{t('bodyProfile.fitScore')}…</Text>
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.card, styles.errorCard]}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  if (!assessment) {
    return (
      <View style={[styles.card, styles.placeholderCard]}>
        <Text style={styles.placeholderText}>
          {t('bodyProfile.openFromTryOn')}
        </Text>
        {onSetupBodyProfile ? (
          <Text style={styles.linkText} onPress={onSetupBodyProfile}>
            {t('bodyProfile.title')} →
          </Text>
        ) : null}
      </View>
    );
  }

  return (
    <View style={styles.card}>
      {/* Overall badge + confidence */}
      <View style={styles.headerRow}>
        <OverallBadge overall={assessment.overall} />
        <ConfidenceChip confidence={assessment.confidence} />
      </View>

      {/* Size recommendation */}
      {assessment.sizeRecommendation && (
        <View style={styles.recommendBox}>
          <Text style={styles.recommendText}>
            {t('bodyProfile.recommendSize', { size: assessment.sizeRecommendation.recommendedSize })}
          </Text>
          <Text style={styles.recommendReason}>{assessment.sizeRecommendation.reason}</Text>
        </View>
      )}

      {/* Per-zone notes */}
      {assessment.zones.length > 0 && (
        <View style={styles.zoneList}>
          {assessment.zones.map((z) => (
            <ZoneRow key={z.zone} zone={z} />
          ))}
        </View>
      )}

      {/* Engine footer — small print, useful for debugging */}
      <Text style={styles.engineFooter}>
        {assessment.engineVersion} · {new Date(assessment.generatedAt).toLocaleTimeString()}
      </Text>
    </View>
  );
};

// ─── Sub-components ──────────────────────────────────────────────────────────

const OverallBadge: React.FC<{ overall: OverallFit }> = ({ overall }) => {
  const { t } = useTranslation();
  const labelKey = (() => {
    switch (overall) {
      case 'good_fit': return 'goodFit';
      case 'tight': case 'too_small': return 'tight';
      case 'relaxed': return 'relaxed';
      case 'oversized': case 'too_large': return 'oversized';
      default: return 'goodFit';
    }
  })();
  const palette = (() => {
    switch (overall) {
      case 'good_fit': return { bg: '#E6F8EE', fg: '#0F7A3A' };
      case 'tight': case 'too_small': return { bg: '#FDEDE8', fg: '#C1351B' };
      case 'relaxed': return { bg: '#FFF5E0', fg: '#A06A00' };
      case 'oversized': case 'too_large': return { bg: '#FDEDE8', fg: '#C1351B' };
      default: return { bg: '#F0F0F0', fg: '#666' };
    }
  })();
  return (
    <View style={[styles.badge, { backgroundColor: palette.bg }]}>
      <Text style={[styles.badgeText, { color: palette.fg }]}>
        {t(`bodyProfile.${labelKey}`)}
      </Text>
    </View>
  );
};

const ConfidenceChip: React.FC<{ confidence: Confidence }> = ({ confidence }) => {
  const { t } = useTranslation();
  const palette = (() => {
    switch (confidence) {
      case 'high': return { bg: '#E6F0FF', fg: '#0055FF' };
      case 'medium': return { bg: '#F0F0F0', fg: '#555' };
      case 'low': return { bg: '#F4F4F4', fg: '#999' };
      default: return { bg: '#F4F4F4', fg: '#999' };
    }
  })();
  return (
    <View style={[styles.chip, { backgroundColor: palette.bg }]}>
      <Text style={[styles.chipText, { color: palette.fg }]}>
        {t('bodyProfile.confidence')}: {t(`bodyProfile.${confidence}`)}
      </Text>
    </View>
  );
};

const ZoneRow: React.FC<{ zone: FitZoneAssessment }> = ({ zone }) => {
  const { glyph, palette } = visualFor(zone.status);
  return (
    <View style={styles.zoneRow}>
      <Text style={[styles.zoneGlyph, { color: palette }]}>{glyph}</Text>
      <View style={styles.zoneTextCol}>
        <Text style={styles.zoneMessage}>{zone.message}</Text>
        {zone.deltaCm != null && (
          <Text style={styles.zoneDelta}>
            {zone.deltaCm > 0 ? '+' : ''}{zone.deltaCm} cm
          </Text>
        )}
      </View>
    </View>
  );
};

function visualFor(status: ZoneStatus): { glyph: string; palette: string } {
  switch (status) {
    case 'good': return { glyph: '✓', palette: '#0F7A3A' };
    case 'snug':
    case 'loose':
    case 'too_short':
    case 'too_long':
      return { glyph: '⚠', palette: '#A06A00' };
    case 'too_tight':
    case 'too_loose':
      return { glyph: '✕', palette: '#C1351B' };
    case 'unknown':
    default:
      return { glyph: '·', palette: '#999' };
  }
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#F8F9FB',
    borderRadius: 14,
    padding: 16,
    marginVertical: 8,
  },
  errorCard: { backgroundColor: '#FDEDE8' },
  placeholderCard: { backgroundColor: '#F0F4FF' },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  loadingText: { marginLeft: 8, color: '#666', fontSize: 14 },
  errorText: { color: '#C1351B', fontSize: 14 },
  placeholderText: { color: '#0055FF', fontSize: 14, marginBottom: 4 },
  linkText: { color: '#0055FF', fontSize: 14, fontWeight: '600', marginTop: 4 },
  badge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  badgeText: { fontSize: 15, fontWeight: '700' },
  chip: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  chipText: { fontSize: 11, fontWeight: '500' },
  recommendBox: {
    backgroundColor: '#E6F0FF',
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  recommendText: { color: '#0055FF', fontSize: 15, fontWeight: '600' },
  recommendReason: { color: '#0055FF', fontSize: 12, marginTop: 2, opacity: 0.8 },
  zoneList: { marginTop: 10, gap: 8 },
  zoneRow: { flexDirection: 'row', alignItems: 'flex-start' },
  zoneGlyph: { fontSize: 16, fontWeight: '700', width: 22, marginTop: 1 },
  zoneTextCol: { flex: 1 },
  zoneMessage: { color: '#222', fontSize: 13, lineHeight: 18 },
  zoneDelta: { color: '#888', fontSize: 11, marginTop: 2 },
  engineFooter: { color: '#B0B0B0', fontSize: 10, marginTop: 10, textAlign: 'right' },
});

export default FitPanel;
