/**
 * BodyProfileScreen — let the user create / edit their calibrated body.
 *
 * Month 1 of docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md. This screen
 * captures the data that the fit engine and (future) SAM 3D Body photo
 * analysis both consume. The MVP flow is manual entry; the photo upload
 * for SAM lands in Month 3.
 *
 * Sections:
 *   1. Height (with source choice: Apple Measure vs manual)
 *   2. Weight (optional)
 *   3. Body Type (chips, from existing mannequin3D.ts BODY_TYPES)
 *   4. Gender (4 options)
 *   5. Optional measurements (chest, waist, hips, shoulders, arm, inseam, foot)
 *   6. Privacy toggles (retain source photo, retain mesh)
 *
 * Persisted via bodyProfileStore (Zustand + AsyncStorage). On save, we
 * also PATCH /body-profiles/:id so the server has a copy.
 *
 * Navigation:
 *   - Reachable from Profile tab (added in Month 1 nav wiring) and
 *     from the try-on screen ("Set up body profile" CTA in Month 2).
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  Pressable,
  Switch,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { useNavigation } from '@react-navigation/native';
import type { StackNavigationProp } from '@react-navigation/stack';

import { useBodyProfileStore, selectActiveProfile } from '../store/bodyProfileStore';
import { BODY_TYPES, BodyTypeId } from '../features/try-on/utils/mannequin3D';
import type { BodyProfile, GenderOption, BodyProfileSource } from '../src/types/bodyProfile';
import type { RootStackParamList } from '../navigation/types';
import useAuthStore from '../store/auth';
import { colors as themeColors } from '../src/theme';

const GENDERS: GenderOption[] = ['male', 'female', 'other', 'prefer_not_to_say'];
const HEIGHT_SOURCES: { id: BodyProfileSource; key: string }[] = [
  { id: 'apple_measure', key: 'heightSourceApple' },
  { id: 'manual', key: 'heightSourceManual' },
];

const MEASUREMENT_FIELDS = [
  { key: 'shoulderWidth', labelKey: 'shoulderWidth' },
  { key: 'chest', labelKey: 'chest' },
  { key: 'waist', labelKey: 'waist' },
  { key: 'hips', labelKey: 'hips' },
  { key: 'armLength', labelKey: 'armLength' },
  { key: 'inseam', labelKey: 'inseam' },
  { key: 'footLength', labelKey: 'footLength' },
] as const;

type MeasurementKey = (typeof MEASUREMENT_FIELDS)[number]['key'];

const BodyProfileScreen: React.FC = () => {
  const { t } = useTranslation();
  const nav = useNavigation<StackNavigationProp<RootStackParamList>>();
  const user = useAuthStore((s: ReturnType<typeof useAuthStore.getState>) => s.user);

  const activeProfile = useBodyProfileStore(selectActiveProfile);
  const createProfile = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.createProfile);
  const updateProfile = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.updateProfile);
  const setHeight = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.setHeight);
  const setWeight = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.setWeight);
  const setBodyType = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.setBodyType);
  const setGender = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.setGender);
  const setMeasurement = useBodyProfileStore((s: ReturnType<typeof useBodyProfileStore.getState>) => s.setMeasurement);

  const [heightCm, setHeightCmLocal] = useState('175');
  const [weight, setWeightLocal] = useState('70');
  const [bodyType, setBodyTypeLocal] = useState<BodyTypeId>('average');
  const [gender, setGenderLocal] = useState<GenderOption>('male');
  const [heightSource, setHeightSourceLocal] = useState<BodyProfileSource>('manual');
  const [measurements, setMeasurements] = useState<Record<MeasurementKey, string>>({
    shoulderWidth: '',
    chest: '',
    waist: '',
    hips: '',
    armLength: '',
    inseam: '',
    footLength: '',
  });
  const [retainPhoto, setRetainPhoto] = useState(false);
  const [retainMesh, setRetainMesh] = useState(true);
  const [saving, setSaving] = useState(false);

  // Hydrate local state from active profile
  useEffect(() => {
    if (!activeProfile) return;
    setHeightCmLocal(String(activeProfile.height.valueCm));
    setWeightLocal(activeProfile.weightKg != null ? String(activeProfile.weightKg) : '70');
    setBodyTypeLocal((activeProfile.bodyType ?? 'average') as BodyTypeId);
    setGenderLocal((activeProfile.gender ?? 'male') as GenderOption);
    setHeightSourceLocal(activeProfile.height.source);
    const m = activeProfile.measurements || {};
    setMeasurements({
      shoulderWidth: m.shoulderWidth ? String(m.shoulderWidth.valueCm) : '',
      chest: m.chest ? String(m.chest.valueCm) : '',
      waist: m.waist ? String(m.waist.valueCm) : '',
      hips: m.hips ? String(m.hips.valueCm) : '',
      armLength: m.armLength ? String(m.armLength.valueCm) : '',
      inseam: m.inseam ? String(m.inseam.valueCm) : '',
      footLength: m.footLength ? String(m.footLength.valueCm) : '',
    });
    setRetainPhoto(activeProfile.privacy?.retainSourcePhoto ?? false);
    setRetainMesh(activeProfile.privacy?.retainMesh ?? true);
  }, [activeProfile?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const onSave = useCallback(async () => {
    if (!user?.id) {
      Alert.alert(t('common.error'), 'Not signed in');
      return;
    }
    const h = parseFloat(heightCm);
    const w = parseFloat(weight);
    if (Number.isNaN(h) || h < 80 || h > 250) {
      Alert.alert(t('common.error'), 'Height must be between 80 and 250 cm');
      return;
    }
    if (Number.isNaN(w) || w < 20 || w > 300) {
      Alert.alert(t('common.error'), 'Weight must be between 20 and 300 kg');
      return;
    }

    setSaving(true);
    try {
      let profile: BodyProfile | undefined = activeProfile;
      if (!profile) {
        profile = createProfile(user.id);
      }
      setHeight(profile.id, h, heightSource);
      setWeight(profile.id, w);
      setBodyType(profile.id, bodyType);
      setGender(profile.id, gender);
      for (const field of MEASUREMENT_FIELDS) {
        const v = measurements[field.key];
        if (v && !Number.isNaN(parseFloat(v))) {
          setMeasurement(profile.id, field.key, parseFloat(v));
        }
      }
      updateProfile(profile.id, {
        privacy: { retainSourcePhoto: retainPhoto, retainMesh },
      });

      // Mirror to the server (best-effort, don't block the user).
      try {
        const apiBase = (process.env.EXPO_PUBLIC_API_URL || '').replace(/\/$/, '');
        if (apiBase) {
          await fetch(`${apiBase}/body-profiles`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${useAuthStore.getState().session?.access_token ?? ''}`,
            },
            body: JSON.stringify({
              height: { valueCm: h, source: heightSource },
              weightKg: w,
              bodyType,
              gender,
              privacy: { retainSourcePhoto: retainPhoto, retainMesh },
            }),
          });
        }
      } catch (serverErr) {
        // Local save already succeeded; don't fail the user on a network blip.
        console.warn('[BodyProfile] server sync failed:', serverErr);
      }

      Alert.alert(t('bodyProfile.saved'));
      nav.goBack();
    } catch (err: any) {
      Alert.alert(t('common.error'), err?.message ?? 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [
    user, activeProfile, heightCm, weight, heightSource, bodyType, gender,
    measurements, retainPhoto, retainMesh, t, nav,
    createProfile, setHeight, setWeight, setBodyType, setGender, setMeasurement, updateProfile,
  ]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>{t('bodyProfile.title')}</Text>
      <Text style={styles.subtitle}>{t('bodyProfile.subtitle')}</Text>

      {/* ── Height ────────────────────────────────────────────────────── */}
      <Text style={styles.sectionLabel}>{t('bodyProfile.height')}</Text>
      <TextInput
        style={styles.input}
        value={heightCm}
        keyboardType="numeric"
        onChangeText={setHeightCmLocal}
        placeholder="175"
        placeholderTextColor={'#9CA3AF'}
      />
      <Text style={styles.hint}>{t('bodyProfile.heightHint')}</Text>

      <Text style={[styles.sectionLabel, styles.spacedLabel]}>
        {t('bodyProfile.heightSource')}
      </Text>
      <View style={styles.chipRow}>
        {HEIGHT_SOURCES.map((src) => {
          const isActive = heightSource === src.id;
          return (
            <Pressable
              key={src.id}
              accessibilityRole="button"
              accessibilityState={{ selected: isActive }}
              onPress={() => setHeightSourceLocal(src.id)}
              style={[styles.chip, isActive && styles.chipActive]}
            >
              <Text style={[styles.chipLabel, isActive && styles.chipLabelActive]}>
                {t(`bodyProfile.${src.key}`)}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* ── Weight ────────────────────────────────────────────────────── */}
      <Text style={[styles.sectionLabel, styles.spacedLabel]}>{t('bodyProfile.weight')}</Text>
      <TextInput
        style={styles.input}
        value={weight}
        keyboardType="numeric"
        onChangeText={setWeightLocal}
        placeholder="70"
        placeholderTextColor={'#9CA3AF'}
      />
      <Text style={styles.hint}>{t('bodyProfile.weightHint')}</Text>

      {/* ── Body type ─────────────────────────────────────────────────── */}
      <Text style={[styles.sectionLabel, styles.spacedLabel]}>{t('bodyProfile.bodyType')}</Text>
      <View style={styles.bodyTypeGrid}>
        {BODY_TYPES.map((bt) => {
          const isActive = bodyType === bt.id;
          return (
            <Pressable
              key={bt.id}
              onPress={() => setBodyTypeLocal(bt.id as BodyTypeId)}
              accessibilityRole="button"
              accessibilityState={{ selected: isActive }}
              style={[styles.btChip, isActive && styles.btChipActive]}
            >
              <Text style={[styles.btLabel, isActive && styles.btLabelActive]}>{bt.label}</Text>
              <Text style={[styles.btDesc, isActive && styles.btDescActive]}>{bt.desc}</Text>
            </Pressable>
          );
        })}
      </View>

      {/* ── Gender ────────────────────────────────────────────────────── */}
      <Text style={[styles.sectionLabel, styles.spacedLabel]}>{t('bodyProfile.gender')}</Text>
      <View style={styles.chipRow}>
        {GENDERS.map((g) => {
          const labelKey =
            g === 'male' ? 'genderMale'
            : g === 'female' ? 'genderFemale'
            : g === 'other' ? 'genderOther'
            : 'genderPreferNot';
          const isActive = gender === g;
          return (
            <Pressable
              key={g}
              onPress={() => setGenderLocal(g)}
              accessibilityRole="button"
              accessibilityState={{ selected: isActive }}
              style={[styles.chip, isActive && styles.chipActive]}
            >
              <Text style={[styles.chipLabel, isActive && styles.chipLabelActive]}>
                {t(`bodyProfile.${labelKey}`)}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* ── Measurements (optional) ───────────────────────────────────── */}
      <Text style={[styles.sectionLabel, styles.spacedLabel]}>
        {t('bodyProfile.measurements')}
      </Text>
      <Text style={styles.hint}>{t('bodyProfile.measurementsHint')}</Text>
      <View style={styles.measurementGrid}>
        {MEASUREMENT_FIELDS.map((field) => (
          <View key={field.key} style={styles.measurementCell}>
            <Text style={styles.measurementLabel}>
              {t(`bodyProfile.${field.labelKey}`)} (cm)
            </Text>
            <TextInput
              style={styles.input}
              value={measurements[field.key]}
              keyboardType="numeric"
              onChangeText={(v) =>
                setMeasurements((prev) => ({ ...prev, [field.key]: v }))
              }
              placeholder="—"
              placeholderTextColor={'#9CA3AF'}
            />
          </View>
        ))}
      </View>

      {/* ── Privacy ───────────────────────────────────────────────────── */}
      <Text style={[styles.sectionLabel, styles.spacedLabel]}>
        {t('bodyProfile.privacyTitle')}
      </Text>
      <View style={styles.privacyRow}>
        <View style={{ flex: 1, paddingRight: 12 }}>
          <Text style={styles.privacyLabel}>{t('bodyProfile.privacyRetainPhoto')}</Text>
          <Text style={styles.hint}>{t('bodyProfile.privacyRetainPhotoHint')}</Text>
        </View>
        <Switch
          value={retainPhoto}
          onValueChange={setRetainPhoto}
          accessibilityLabel={t('bodyProfile.privacyRetainPhoto')}
        />
      </View>
      <View style={styles.privacyRow}>
        <View style={{ flex: 1, paddingRight: 12 }}>
          <Text style={styles.privacyLabel}>{t('bodyProfile.privacyRetainMesh')}</Text>
        </View>
        <Switch
          value={retainMesh}
          onValueChange={setRetainMesh}
          accessibilityLabel={t('bodyProfile.privacyRetainMesh')}
        />
      </View>

      {/* ── Save ──────────────────────────────────────────────────────── */}
      <Pressable
        style={[styles.saveBtn, saving && styles.saveBtnDisabled]}
        onPress={onSave}
        disabled={saving}
        accessibilityRole="button"
      >
        {saving
          ? <ActivityIndicator color="#fff" />
          : <Text style={styles.saveLabel}>{t('bodyProfile.save')}</Text>}
      </Pressable>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 60 },
  title: { fontSize: 26, fontWeight: '700', marginBottom: 6 },
  subtitle: { fontSize: 14, color: '#666', marginBottom: 22 },
  sectionLabel: { fontSize: 15, fontWeight: '600', marginBottom: 6 },
  spacedLabel: { marginTop: 18 },
  input: {
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    backgroundColor: '#F8F8F8',
  },
  hint: { fontSize: 12, color: '#888', marginTop: 4 },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F0F0F0',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  chipActive: { backgroundColor: '#0055FF', borderColor: '#0055FF' },
  chipLabel: { color: '#333', fontSize: 14, fontWeight: '500' },
  chipLabelActive: { color: '#fff' },
  bodyTypeGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  btChip: {
    flexBasis: '48%',
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#F0F0F0',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  btChipActive: { backgroundColor: '#E0EBFF', borderColor: '#0055FF' },
  btLabel: { fontSize: 15, fontWeight: '600', color: '#222' },
  btLabelActive: { color: '#0055FF' },
  btDesc: { fontSize: 12, color: '#666', marginTop: 2 },
  btDescActive: { color: '#0055FF' },
  measurementGrid: { marginTop: 8, gap: 10 },
  measurementCell: {},
  measurementLabel: { fontSize: 13, color: '#444', marginBottom: 4 },
  privacyRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 10 },
  privacyLabel: { fontSize: 14, fontWeight: '500' },
  saveBtn: {
    backgroundColor: '#0055FF',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 28,
  },
  saveBtnDisabled: { opacity: 0.6 },
  saveLabel: { color: '#fff', fontSize: 16, fontWeight: '600' },
});

export default BodyProfileScreen;
