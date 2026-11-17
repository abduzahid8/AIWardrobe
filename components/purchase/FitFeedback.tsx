import React, { useEffect, useState, useRef } from 'react';
import { View, TouchableOpacity, StyleSheet, TextInput } from 'react-native';
import { ScaledText } from '../ui/ScaledText';
import { usePurchaseIntelligenceStore } from '../../store/purchaseIntelligenceStore';
import type { FitOutcome, FitZone, ZoneOutcome } from '../../src/types/purchase';
export function FitFeedback({
  productId,
  ownerId,
  brand,
  category,
  sizeLabel,
  subcategory,
}: {
  productId: string;
  ownerId?: string;
  brand?: string;
  category: string;
  sizeLabel?: string;
  subcategory?: string;
}) {
  const answerBusy = useRef(false);
  const store = usePurchaseIntelligenceStore(),
    [saved, setSaved] = useState<string>(),
    [details, setDetails] = useState(false),
    [zone, setZone] = useState<FitZone>(),
    [error, setError] = useState(false),
    [size, setSize] = useState(sizeLabel || ''),
    [finished, setFinished] = useState(false),
    [manualOpen, setManualOpen] = useState(false);
  useEffect(() => {
    void store.hydrate().catch(() => setError(true));
  }, [store.hydrate]);
  useEffect(() => {
    setSaved(undefined);
    setDetails(false);
    setZone(undefined);
    setSize(sizeLabel || '');
    setFinished(false);
    setError(false);
  }, [productId, sizeLabel]);
  if (
    (ownerId && ownerId !== store.ownerId) ||
    finished ||
    !store.ownerId ||
    !store.feedbackEnabled
  )
    return null;
  if (!saved && !manualOpen && !store.canPrompt(productId)) {
    const prior = store.evidence.filter((e) => e.productId === productId).at(-1);
    return prior ? (
      <TouchableOpacity
        accessibilityRole="button"
        style={styles.button}
        onPress={() => {
          setSaved(prior.id);
          setSize(prior.sizeLabel || '');
        }}
      >
        <ScaledText>Update fit / reference</ScaledText>
      </TouchableOpacity>
    ) : null;
  }
  const answer = async (overallFit: FitOutcome) => {
    if (answerBusy.current) return;
    answerBusy.current = true;
    try {
      const evidence = await store.recordFit({
        productId,
        brand,
        category,
        subcategory,
        sizeLabel: size.trim() || undefined,
        overallFit,
        source: 'owned',
        confidence: 0.8,
      });
      setSaved(evidence.id);
      setManualOpen(false);
    } catch {
      setError(true);
    } finally {
      answerBusy.current = false;
    }
  };
  const zones: FitZone[] = [
    'shoulders',
    'chest',
    'waist',
    'hips',
    'sleeve',
    'thigh',
    'inseam',
    'shoe_length',
    'shoe_width',
  ];
  return (
    <View style={styles.card}>
      <ScaledText>{saved ? 'Fit feedback saved on this device' : 'How does this fit?'}</ScaledText>
      {!saved ? (
        <>
          <View style={styles.row}>
            {(
              [
                ['too_small', 'Too small'],
                ['snug', 'Snug'],
                ['perfect', 'Perfect'],
                ['relaxed', 'Relaxed'],
                ['too_large', 'Too big'],
              ] as const
            ).map(([value, label]) => (
              <TouchableOpacity
                accessibilityRole="button"
                accessibilityLabel={label}
                key={value}
                style={styles.button}
                onPress={() => void answer(value)}
              >
                <ScaledText>{label}</ScaledText>
              </TouchableOpacity>
            ))}
          </View>
          <TouchableOpacity
            accessibilityRole="button"
            onPress={() => void store.dismiss(productId).catch(() => setError(true))}
          >
            <ScaledText>Not now</ScaledText>
          </TouchableOpacity>
        </>
      ) : (
        <>
          <TouchableOpacity
            accessibilityRole="button"
            onPress={() => {
              setSaved(undefined);
              setManualOpen(true);
            }}
          >
            <ScaledText>Change fit answer</ScaledText>
          </TouchableOpacity>
          <TextInput
            accessibilityLabel="Garment size"
            placeholder="Size (optional)"
            value={size}
            onChangeText={setSize}
            maxLength={24}
          />
          <TouchableOpacity accessibilityRole="button" onPress={() => setDetails(!details)}>
            <ScaledText>{details ? 'Close details' : 'Add optional details'}</ScaledText>
          </TouchableOpacity>
          {brand && (
            <TouchableOpacity
              accessibilityRole="button"
              disabled={
                !size.trim() ||
                !['perfect', 'snug', 'relaxed'].includes(
                  store.evidence.find((e) => e.id === saved)?.overallFit || ''
                )
              }
              accessibilityState={{
                disabled:
                  !size.trim() ||
                  !['perfect', 'snug', 'relaxed'].includes(
                    store.evidence.find((e) => e.id === saved)?.overallFit || ''
                  ),
              }}
              onPress={() =>
                void store
                  .updateFit(saved, { reference: true, sizeLabel: size.trim() || undefined })
                  .catch(() => setError(true))
              }
            >
              <ScaledText>Use as fit reference</ScaledText>
            </TouchableOpacity>
          )}
          {details && (
            <>
              <ScaledText>Preferred fit (optional)</ScaledText>
              <View style={styles.row}>
                {(['snug', 'regular', 'relaxed'] as const).map((value) => (
                  <TouchableOpacity
                    accessibilityRole="button"
                    key={value}
                    style={styles.button}
                    onPress={() =>
                      void store
                        .updateFit(saved, { preferredFit: value })
                        .catch(() => setError(true))
                    }
                  >
                    <ScaledText>{`Prefer ${value}`}</ScaledText>
                  </TouchableOpacity>
                ))}
              </View>
              <View style={styles.row}>
                {zones.map((value) => (
                  <TouchableOpacity
                    key={value}
                    accessibilityRole="button"
                    accessibilityState={{ selected: zone === value }}
                    style={styles.button}
                    onPress={() => setZone(value)}
                  >
                    <ScaledText>{value.replace(/_/g, ' ')}</ScaledText>
                  </TouchableOpacity>
                ))}
              </View>
              {zone && (
                <View style={styles.row}>
                  {(['tight', 'good', 'loose', 'short', 'long'] as ZoneOutcome[]).map((value) => (
                    <TouchableOpacity
                      accessibilityRole="button"
                      key={value}
                      style={styles.button}
                      onPress={() => {
                        const prior = store.evidence.find((e) => e.id === saved);
                        void store
                          .updateFit(saved, { zones: { ...prior?.zones, [zone]: value } })
                          .catch(() => setError(true));
                      }}
                    >
                      <ScaledText>{value}</ScaledText>
                    </TouchableOpacity>
                  ))}
                </View>
              )}
            </>
          )}
          <TouchableOpacity accessibilityRole="button" onPress={() => setFinished(true)}>
            <ScaledText>Done</ScaledText>
          </TouchableOpacity>
        </>
      )}
      {error && (
        <ScaledText>Could not save feedback. You can continue using your wardrobe.</ScaledText>
      )}
    </View>
  );
}
const styles = StyleSheet.create({
  card: { padding: 16, marginVertical: 12, borderRadius: 16, backgroundColor: '#F0F4FA', gap: 10 },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  button: { padding: 10, backgroundColor: '#fff', borderRadius: 10 },
});
