import React, { useState } from 'react';
import { Modal, Platform, Pressable, StyleSheet, TouchableOpacity, View,  } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
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

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

interface LanguageSwitcherProps {
  visible: boolean;
  onClose: () => void;
}

const LanguageSwitcher: React.FC<LanguageSwitcherProps> = ({ visible, onClose }) => {
  const { t } = useTranslation();
  const { currentLanguage, setLanguage } = useLanguageStore();
  const { colors, isDark } = useTheme();
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

  const tint = isDark ? 'dark' : 'light';
  const glass = isDark ? 'rgba(17, 20, 30, 0.58)' : 'rgba(255, 255, 255, 0.56)';
  const glassBorder = isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(255, 255, 255, 0.68)';
  const text = colors.text.primary;
  const textSub = colors.text.secondary;
  const accent = isDark ? '#A8C0DA' : '#12385F';
  const accentStart = isDark ? '#446B95' : '#2A537F';
  const accentEnd = isDark ? '#1C3654' : '#0D2743';

  const LanguageOption: React.FC<{ lang: Language }> = ({ lang }) => {
    const isSelected = lang === currentLanguage;
    const isLoadingLang = loading === lang;

    return (
      <AnimatedPressable
        entering={FadeIn}
        onPress={() => handleLanguageSelect(lang)}
        style={({ pressed }) => [
          styles.languageOption,
          pressed && styles.languageOptionPressed,
          isSelected && styles.languageOptionSelected,
          { borderColor: isSelected ? accent : glassBorder },
        ]}
      >
        <BlurView
          intensity={Platform.OS === 'ios' ? (isSelected ? 60 : 40) : 100}
          tint={tint}
          style={StyleSheet.absoluteFillObject}
        />
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
              <LinearGradient
                colors={['rgba(255,255,255,0.3)', 'rgba(255,255,255,0.1)']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.checkBackground}
              >
                <Ionicons name="checkmark-circle" size={24} color="#FFFFFF" />
              </LinearGradient>
            </View>
          ) : (
            <Ionicons name="chevron-forward" size={20} color={textSub} />
          )}
        </View>
      </AnimatedPressable>
    );
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={onClose}
    >
      <Animated.View
        entering={FadeIn}
        exiting={FadeOut}
        style={styles.overlay}
      >
        <Pressable style={styles.overlayPressable} onPress={onClose}>
          <BlurView
            intensity={Platform.OS === 'ios' ? 60 : 100}
            tint="dark"
            style={StyleSheet.absoluteFillObject}
          />
        </Pressable>

        <Animated.View
          entering={SlideInDown.duration(300).damping(28)}
          exiting={SlideOutDown.duration(300).damping(28)}
          style={styles.container}
        >
          <View style={[styles.content, { backgroundColor: glass }]}>
            <BlurView
              intensity={Platform.OS === 'ios' ? 50 : 100}
              tint={tint}
              style={StyleSheet.absoluteFillObject}
            />

            <View style={styles.header}>
              <ScaledText style={[styles.title, { color: text }]}>
                {t('language.selectLanguage')}
              </ScaledText>
              <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                <BlurView
                  intensity={Platform.OS === 'ios' ? 30 : 100}
                  tint={tint}
                  style={StyleSheet.absoluteFillObject}
                />
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
                {t('language.selectLanguage')}
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
  },
  overlayPressable: {
    flex: 1,
  },
  container: {
    justifyContent: 'flex-end',
  },
  content: {
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    paddingTop: 24,
    paddingBottom: 40,
    paddingHorizontal: 20,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    letterSpacing: -0.5,
  },
  closeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
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
  },
  footer: {
    marginTop: 24,
    paddingTop: 16,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(128, 128, 128, 0.2)',
    alignItems: 'center',
  },
  footerText: {
    fontSize: 13,
    fontWeight: '500',
  },
});

export default LanguageSwitcher;
