import React, { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Alert, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useTranslation } from 'react-i18next';
import { authEvents } from '../../store/authEvents';
import useUploadQueueStore from '../../src/store/uploadQueueStore';
import processor from '../../src/services/uploadQueueProcessor';

function subscribeIdentity(changed: () => void) {
  const offLogin = authEvents.onLogin(changed), offLogout = authEvents.onLogout(changed);
  return () => { offLogin(); offLogout(); };
}
export function PendingScanQueue({ disabled = false }: { disabled?: boolean }) {
  const { t } = useTranslation();
  const owner = useSyncExternalStore(subscribeIdentity, authEvents.getUserId, authEvents.getUserId);
  const uploads = useUploadQueueStore(state => state.pendingUploads);
  const entries = uploads.filter(entry => !!owner && entry.ownerId === owner);
  const [busy, setBusy] = useState<string | null>(null);
  const mounted = useRef(true), active = useRef(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { setBusy(null); }, [owner]);
  const run = async (id: string, discard: boolean) => {
    if (disabled || active.current || !owner || owner !== authEvents.getUserId()) return;
    const revision = authEvents.getRevision();
    const current = () => mounted.current && owner === authEvents.getUserId() && revision === authEvents.getRevision();
    active.current = true; setBusy(id);
    try {
      const accepted = discard ? await processor.discardUpload(id) : await processor.retryUpload(id);
      if (!accepted && current()) Alert.alert(t('common.error'), t('wardrobeVideo.queueBusy'));
    } catch {
      if (current()) Alert.alert(t('common.error'), t('wardrobeVideo.queueActionFailed'));
    } finally {
      active.current = false;
      if (current()) setBusy(null);
    }
  };
  if (!entries.length) return null;
  return <View style={styles.panel}>
    <Text style={styles.title}>{t('wardrobeVideo.pendingScans')}</Text>
    <Text>{t('wardrobeVideo.pendingScansDescription')}</Text>
    {entries.map((entry, index) => {
      const label = t(entry.type === 'video' ? 'wardrobeVideo.queuedVideo' : 'wardrobeVideo.queuedPhoto', { index:index + 1 });
      const locked = disabled || !!busy || entry.status === 'processing';
      return <View key={entry.id} style={styles.row}>
        <Text style={styles.label}>{label} · {t(`wardrobeVideo.queueStatus_${entry.status}`)}</Text>
        <View style={styles.actions}>
          <TouchableOpacity accessibilityRole="button" accessibilityLabel={`${t('wardrobeVideo.retryScan')}: ${label}`}
            disabled={locked} onPress={() => { void run(entry.id, false); }} style={locked ? styles.disabled : undefined}>
            <Text style={styles.action}>{t('wardrobeVideo.retryScan')}</Text>
          </TouchableOpacity>
          <TouchableOpacity accessibilityRole="button" accessibilityLabel={`${t('wardrobeVideo.removeScan')}: ${label}`}
            disabled={locked} style={locked ? styles.disabled : undefined}
            onPress={() => Alert.alert(t('wardrobeVideo.removeScan'), t('wardrobeVideo.removeScanConfirmation'), [
              { text:t('common.cancel'), style:'cancel' },
              { text:t('wardrobeVideo.removeScan'), style:'destructive', onPress:() => { void run(entry.id, true); } },
            ])}>
            <Text style={styles.action}>{t('wardrobeVideo.removeScan')}</Text>
          </TouchableOpacity>
        </View>
      </View>;
    })}
  </View>;
}
const styles = StyleSheet.create({
  panel:{ marginBottom:20, padding:16, borderRadius:16, backgroundColor:'#FFFFFF' },
  title:{ fontWeight:'600', fontSize:17, marginBottom:6 }, row:{ paddingTop:14 },
  label:{ marginBottom:8 }, actions:{ flexDirection:'row', gap:24 },
  action:{ color:'#0066CC', paddingVertical:8 }, disabled:{ opacity:0.4 },
});
