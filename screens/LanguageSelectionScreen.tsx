import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ScaledText } from '../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import useLanguageStore, { SUPPORTED_LANGUAGES, type Language } from '../store/languageStore';
import type { RootStackParamList } from '../navigation/types';

const LANGUAGE_OPTIONS: { code: Language; flag: string; nativeName: string }[] = [
  { code: 'en', flag: '🇬🇧', nativeName: 'English' },
  { code: 'ru', flag: '🇷🇺', nativeName: 'Русский' },
  { code: 'uz', flag: '🇺🇿', nativeName: "O'zbek" },
];

/**
 * First-run gate: the user can ONLY choose a language here.
 * No skip, no back, no auth, no Continue button — tapping a language
 * persists it via languageStore.setLanguage() and PUSHES SignIn, so the
 * chooser stays natively reachable (iOS edge-swipe / Android back).
 */
const LanguageSelectionScreen = () => {
  const { t } = useTranslation();
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const currentLanguage = useLanguageStore((s) => s.currentLanguage);
  const setLanguage = useLanguageStore((s) => s.setLanguage);
  const isSaving = useLanguageStore((s) => s.isLoading);

  const initial: Language = SUPPORTED_LANGUAGES.includes(currentLanguage)
    ? currentLanguage
    : 'en';
  const [selected, setSelected] = useState<Language>(initial);

  const handleSelect = async (language: Language) => {
    if (isSaving) return;
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Instant checkmark feedback; the store applies the locale AND
    // persists it, then we push forward — the chooser stays underneath
    // for native back. Stack navigate() reuses SignIn if already present,
    // so re-selecting after a swipe-back never stacks duplicates.
    setSelected(language);
    await setLanguage(language);
    navigation.navigate('SignIn');
  };

  return (
    <SafeAreaView style={styles.safe} testID="languageSelectionScreen">
      <View style={styles.container}>
        <View style={styles.header}>
          <ScaledText style={styles.title} testID="languageSelectionTitle">
            {t('language.chooseTitle', { defaultValue: 'Choose your language' })}
          </ScaledText>
          <ScaledText style={styles.subtitle}>
            {t('language.chooseSubtitle', {
              defaultValue: 'You can change it later in settings.',
            })}
          </ScaledText>
        </View>

        <View style={styles.options}>
          {LANGUAGE_OPTIONS.map((option) => {
            const isActive = selected === option.code;
            const showSpinner = isActive && isSaving;
            return (
              <TouchableOpacity
                key={option.code}
                testID={`languageOption-${option.code}`}
                accessibilityRole="button"
                accessibilityState={{ selected: isActive }}
                accessibilityLabel={option.nativeName}
                activeOpacity={0.85}
                onPress={() => void handleSelect(option.code)}
                disabled={isSaving}
                style={[styles.card, isActive && styles.cardActive]}
              >
                <View style={styles.cardLeft}>
                  <ScaledText style={styles.flag}>{option.flag}</ScaledText>
                  <View>
                    <ScaledText style={styles.nativeName}>{option.nativeName}</ScaledText>
                    <ScaledText style={styles.codeLabel}>
                      {option.code.toUpperCase()}
                    </ScaledText>
                  </View>
                </View>
                {showSpinner ? (
                  <ActivityIndicator size="small" color="#0A1931" />
                ) : isActive ? (
                  <Ionicons name="checkmark-circle" size={26} color="#0A1931" />
                ) : (
                  <Ionicons name="ellipse-outline" size={26} color="#CBD5E1" />
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 48,
    paddingBottom: 32,
    justifyContent: 'center',
    maxWidth: 480,
    width: '100%',
    alignSelf: 'center',
  },
  header: {
    marginBottom: 32,
    alignItems: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: '800',
    color: '#0A1931',
    textAlign: 'center',
    marginBottom: 10,
    letterSpacing: 0.3,
  },
  subtitle: {
    fontSize: 15,
    color: '#64748B',
    textAlign: 'center',
    lineHeight: 22,
  },
  options: {
    gap: 12,
    marginBottom: 8,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#F8FAFC',
    borderWidth: 2,
    borderColor: '#E2E8F0',
    borderRadius: 18,
    paddingVertical: 16,
    paddingHorizontal: 18,
  },
  cardActive: {
    backgroundColor: '#EEF4FF',
    borderColor: '#0A1931',
  },
  cardLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  flag: {
    fontSize: 30,
  },
  nativeName: {
    fontSize: 17,
    fontWeight: '700',
    color: '#0A1931',
  },
  codeLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748B',
    letterSpacing: 1,
    marginTop: 2,
  },
});

export default LanguageSelectionScreen;
