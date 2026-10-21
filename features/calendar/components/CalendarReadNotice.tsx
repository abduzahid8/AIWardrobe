import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { useTranslation } from 'react-i18next';
import { ScaledText } from '../../../components/ui/ScaledText';

export function CalendarReadNotice({ onRetry }: { onRetry: () => void }) {
    const { t } = useTranslation();
    return <View accessibilityRole="alert" style={styles.notice}>
        <ScaledText>{t('calendar.loadError')}</ScaledText>
        <TouchableOpacity accessibilityRole="button" accessibilityLabel={t('common.retry')} onPress={onRetry} style={styles.retry}>
            <ScaledText style={styles.label}>{t('common.retry')}</ScaledText>
        </TouchableOpacity>
    </View>;
}
const styles = StyleSheet.create({
    notice: { padding: 16, marginBottom: 16, borderRadius: 12, backgroundColor: '#FFF4DB' },
    retry: { alignSelf: 'flex-start', minHeight: 44, justifyContent: 'center', paddingHorizontal: 8 },
    label: { color: '#0A1931', fontWeight: '600' },
});
