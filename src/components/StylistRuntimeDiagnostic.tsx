import React from 'react';
import {Text} from 'react-native';
import Config from '../config/env';

export const clientRevision = process.env.EXPO_PUBLIC_CLIENT_REVISION || 'unstamped';
export function StylistRuntimeDiagnostic() {
  if (!__DEV__) return null;
  return <Text testID="stylistRuntime" style={{fontSize: 10, textAlign: 'center', color: '#526174'}}>
    {`Client ${clientRevision} · ${new URL(Config.supabase.url).hostname}`}
  </Text>;
}
