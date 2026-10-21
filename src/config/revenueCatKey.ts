/** Public SDK keys identify one store app; never send an Apple key to Google Play. */
export function selectRevenueCatKey(platform: string, keys: { ios?: string; android?: string; legacy?: string }): string {
  if (platform !== 'ios' && platform !== 'android') return '';
  const key = (platform === 'ios' ? keys.ios : keys.android) || keys.legacy || '';
  const prefix = platform === 'ios' ? 'appl_' : 'goog_';
  return key.startsWith(prefix) || key.startsWith('test_') ? key : '';
}
