import React from 'react';
import { StyleSheet, View, TouchableOpacity } from 'react-native';
import { useTranslation } from 'react-i18next';
import { useAppNavigation } from '../hooks/useAppNavigation';
import useWardrobeStore from '../store/wardrobeStore';
import { ScaledText } from './ui/ScaledText';

export function WardrobeCapacityNotice() {
    const { t } = useTranslation();
    const navigation = useAppNavigation();
    const issue = useWardrobeStore(state => state.wardrobeCapacityIssue);
    const syncing = useWardrobeStore(state => state.isSyncing);
    const sync = useWardrobeStore(state => state.syncToServer);
    if (!issue) return null;
    return <View style={styles.notice} accessibilityLiveRegion="polite">
        <ScaledText style={styles.title}>{t('wardrobeSync.capacityTitle')}</ScaledText>
        <ScaledText style={styles.body}>{t('wardrobeSync.capacityMessage', { limit: issue.limit })}</ScaledText>
        <View style={styles.actions}>
            <TouchableOpacity accessibilityRole="button" accessibilityState={{ disabled: syncing }} disabled={syncing}
                onPress={() => { void sync().catch(() => undefined); }} style={styles.button}>
                <ScaledText style={styles.link}>{t(syncing ? 'wardrobeSync.retrying' : 'wardrobeSync.retry')}</ScaledText>
            </TouchableOpacity>
            <TouchableOpacity accessibilityRole="button" onPress={() => navigation.navigate('Paywall')} style={styles.button}>
                <ScaledText style={styles.link}>{t('wardrobeSync.upgrade')}</ScaledText>
            </TouchableOpacity>
        </View>
    </View>;
}
const styles = StyleSheet.create({
    notice: { marginHorizontal: 16, marginBottom: 12, padding: 14, borderRadius: 12, backgroundColor: '#FFF4DD' },
    title: { color: '#593600', fontWeight: '700', fontSize: 15 },
    body: { color: '#593600', fontSize: 13, marginTop: 6 },
    actions: { flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginTop: 8 },
    button: { minHeight: 44, justifyContent: 'center', paddingHorizontal: 4 },
    link: { color: '#593600', fontWeight: '600' },
});
