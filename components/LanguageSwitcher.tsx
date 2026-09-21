import React, { useState } from 'react';
import { Modal, Platform, Pressable, StyleSheet, TouchableOpacity, View, } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';
import { useTranslation } from 'react-i18next';
import * as Haptics from 'expo-haptics';
import useLanguageStore, { Language, SUPPORTED_LANGUAGES, LANGUAGE_NAMES } from '../store/languageStore';
import { useTheme } from '../src/theme/ThemeContext';

interface LanguageSwitcherProps {
  visible: boolean;
  onClose: () => void;
}

const LanguageSwitcher: React.FC<LanguageSwitcherProps> = ({ visible, onClose }) => {
  const { t } = useTranslation();
  const { currentLanguage, setLanguage } = useLanguageStore();
  const { colors, isDark } = useTheme();
  const insets = useSafeAreaInsets();
  const [loading, setLoading] = useState<Language | null>(null);

  const handleLanguageSelect = async (lang: Language) => {
    if (lang === currentLanguage) {
      onClose();
      return;
    }

    setLoading(lang);
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    try {
      await setLanguage(lang);
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      onClose();
    } catch (error) {
      console.error('Failed to change language:', error);
    } finally {
      setLoading(null);
    }
  };

  // Solid, opaque surfaces so the profile list can never bleed through.
  const sheetBg = isDark ? '#171D30' : '#FFFFFF';
  const optionBg = isDark ? 'rgba(255,255,255,0.06)' : '#F2F5F9';
  const optionBorder = isDark ? 'rgba(255,255,255,0.12)' : '#E2E8F0';
  const closeBg = isDark ? 'rgba(255,255,255,0.08)' : '#EEF2F7';
  const text = colors.text.primary;
  const textSub = colors.text.secondary;
  const accent = isDark ? '#A8C0DA' : '#12385F';
  const accentStart = isDark ? '#446B95' : '#2A537F';
  const accentEnd = isDark ? '#1C3654' : '#0D2743';

  const LanguageOption: React.FC<{ lang: Language }> = ({ lang }) => {
    const isSelected = lang === currentLanguage;
    const isLoadingLang = loading === lang;

    return (
      <Pressable
        onPress={() => handleLanguageSelect(lang)}
        style={({ pressed }) => [
          styles.languageOption,
          { backgroundColor: isSelected ? 'transparent' : optionBg },
          { borderColor: isSelected ? accent : optionBorder },
          isSelected && styles.languageOptionSelected,
          pressed && styles.languageOptionPressed,
        ]}
      >
        {isSelected && (
          <LinearGradient
            colors={[accentStart, accentEnd]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={StyleSheet.absoluteFillObject}
          />
        )}

        <View style={styles.languageOptionContent}>
          <View style={styles.languageInfo}>
            <ScaledText
              style={[
                styles.languageName,
                { color: isSelected ? '#FFFFFF' : text },
              ]}
            >
              {LANGUAGE_NAMES[lang]}
            </ScaledText>
            <ScaledText
              style={[
                styles.languageCode,
                { color: isSelected ? 'rgba(255,255,255,0.8)' : textSub },
              ]}
            >
              {lang.toUpperCase()}
            </ScaledText>
          </View>

          {isLoadingLang ? (
            <View style={styles.loadingContainer}>
              <ScaledText style={[styles.loadingText, { color: isSelected ? '#FFFFFF' : accent }]}>
                {t('common.loading')}
              </ScaledText>
            </View>
          ) : isSelected ? (
            <View style={styles.checkContainer}>
              <View style={styles.checkBackground}>
                <Ionicons name="checkmark-circle" size={24} color="#FFFFFF" />
              </View>
            </View>
          ) : (
            <Ionicons name="chevron-forward" size={20} color={textSub} />
          )}
        </View>
      </Pressable>
    );
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <Animated.View
        entering={FadeIn.duration(180)}
        exiting={FadeOut.duration(180)}
        style={styles.overlay}
      >
        <Pressable style={styles.overlayPressable} onPress={onClose} />

        <Animated.View
          entering={SlideInDown.duration(300).damping(28)}
          exiting={SlideOutDown.duration(300).damping(28)}
          style={styles.container}
        >
          <View style={[styles.content, { backgroundColor: sheetBg, paddingBottom: Math.max(insets.bottom, 16) + 24 }]}>
            <View style={styles.handle} />

            <View style={styles.header}>
              <ScaledText style={[styles.title, { color: text }]}>
                {t('language.selectLanguage')}
              </ScaledText>
              <TouchableOpacity
                onPress={onClose}
                style={[styles.closeButton, { backgroundColor: closeBg }]}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons name="close" size={20} color={textSub} />
              </TouchableOpacity>
            </View>

            <View style={styles.languageList}>
              {SUPPORTED_LANGUAGES.map((lang) => (
                <LanguageOption key={lang} lang={lang} />
              ))}
            </View>

            <View style={styles.footer}>
              <ScaledText style={[styles.footerText, { color: textSub }]}>
                {t('language.chooseSubtitle')}
              </ScaledText>
            </View>
          </View>
        </Animated.View>
      </Animated.View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
    // Opaque dim — guarantees the sheet reads as a modal on top of the profile.
    backgroundColor: 'rgba(8, 10, 18, 0.55)',
  },
  overlayPressable: {
    flex: 1,
  },
  container: {
    justifyContent: 'flex-end',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -8 },
        shadowOpacity: 0.25,
        shadowRadius: 24,
      },
      android: {
        elevation: 24,
      },
    }),
  },
  content: {
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    paddingTop: 12,
    paddingHorizontal: 20,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderLeftWidth: StyleSheet.hairlineWidth,
    borderRightWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(128, 128, 128, 0.25)',
  },
  handle: {
    width: 38,
    height: 5,
    borderRadius: 999,
    backgroundColor: 'rgba(128, 128, 128, 0.35)',
    alignSelf: 'center',
    marginBottom: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: -0.5,
    flex: 1,
  },
  closeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 12,
  },
  languageList: {
    gap: 12,
  },
  languageOption: {
    borderRadius: 16,
    borderWidth: 1.5,
    overflow: 'hidden',
    padding: 16,
  },
  languageOptionPressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
  languageOptionSelected: {
    borderWidth: 2,
  },
  languageOptionContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  languageInfo: {
    flex: 1,
  },
  languageName: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  languageCode: {
    fontSize: 14,
    fontWeight: '500',
  },
  loadingContainer: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  loadingText: {
    fontSize: 14,
    fontWeight: '600',
  },
  checkContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    overflow: 'hidden',
  },
  checkBackground: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  footer: {
    marginTop: 20,
    paddingTop: 16,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(128, 128, 128, 0.2)',
    alignItems: 'center',
  },
  footerText: {
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
  },
});

export default LanguageSwitcher;
