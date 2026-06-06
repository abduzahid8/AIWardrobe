import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList } from '../navigation/types';
import { useTranslation } from 'react-i18next';

import { iapService } from '../src/services/iapService';
import * as Haptics from 'expo-haptics';

const TrialExpiredScreen: React.FC = () => {
  const { t } = useTranslation();
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [isRestoring, setIsRestoring] = React.useState(false);

  const handleRestore = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setIsRestoring(true);
    try {
      const result = await iapService.restorePurchases();
      if (result.success) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        // The RootNavigator will automatically react to the state change
        // and remove the TrialExpired screen.
      } else {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
        alert(result.error || 'No previous purchases found.');
      }
    } catch (error) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      alert('Failed to restore purchases. Please try again.');
    } finally {
      setIsRestoring(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.iconContainer}>
          <Ionicons name="time" size={40} color="#7C3AED" />
        </View>
        <ScaledText style={styles.title}>{t('trialExpired.title')}</ScaledText>
        <ScaledText style={styles.body}>{t('trialExpired.body')}</ScaledText>
        <TouchableOpacity style={styles.primaryButton} onPress={() => navigation.navigate('Paywall')}>
          <ScaledText style={styles.primaryButtonText}>{t('trialExpired.upgradeToPro')}</ScaledText>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.secondaryButton} 
          onPress={handleRestore}
          disabled={isRestoring}
        >
          <ScaledText style={styles.secondaryButtonText}>
            {isRestoring ? 'Restoring...' : 'Already a pro? Restore'}
          </ScaledText>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B1020',
  },
  iconContainer: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: 'rgba(124, 58, 237, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    alignSelf: 'center',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 12,
    textAlign: 'center',
  },
  body: {
    fontSize: 16,
    lineHeight: 24,
    color: '#CBD5E1',
    marginBottom: 24,
    textAlign: 'center',
  },
  primaryButton: {
    backgroundColor: '#7C3AED',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginBottom: 14,
    width: '100%',
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 15,
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 12,
  },
  secondaryButtonText: {
    color: '#7C3AED',
    fontWeight: '600',
    fontSize: 14,
  },
});

export default TrialExpiredScreen;
