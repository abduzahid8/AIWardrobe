import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ScaledText } from '../../components/ui/ScaledText';
import { wardrobeMediaReference } from '../services/wardrobeMedia';

/** How long a failed product photo waits before its single automatic retry. */
export const PRODUCT_IMAGE_RETRY_DELAY_MS = 600;

/** Schemes the RN/expo image loaders can actually resolve. Anything else
 * (empty, relative path, garbage) has no photo to show at all. */
const USABLE_URI_RE = /^(https?:|file:|data:|asset:|content:)/i;
/** Only merchant/CDN http(s) photos benefit from a cache-busting re-fetch. */
const BUSTABLE_URI_RE = /^https?:\/\//i;

/** A raw product image string is renderable only with a loader-supported scheme. */
export function usableProductImageUri(raw: unknown): string {
  return typeof raw === 'string' && USABLE_URI_RE.test(raw.trim()) ? raw.trim() : '';
}

/** Fresh cache-busting query param so a transient CDN hiccup re-fetches. */
export function cacheBustedImageUri(uri: string): string {
  const bust = `aiwr=${Date.now()}`;
  return uri.includes('?') ? `${uri}&${bust}` : `${uri}?${bust}`;
}

/** Private wardrobe media is re-signed per request and its authenticated
 * canonical URL drops query params (mediaImageSource -> wardrobeMediaReference),
 * so a cache-busted retry would request the identical URL. Such sources and
 * local file/data URIs fail over to the placeholder on their first error. */
function bustableProductImageUri(uri: string): boolean {
  return BUSTABLE_URI_RE.test(uri) && !wardrobeMediaReference(uri);
}

export type ProductImagePhase = 'ready' | 'unavailable';

export interface ProductImageRetryState {
  phase: ProductImagePhase;
  /** URI to render now ('' when the slot has no usable photo at all). */
  activeUri: string;
  /** Feed expo-image / RN Image onError here. */
  onImageError: () => void;
}

/**
 * Honest product-photo state for slot tiles and product thumbs: ONE bounded
 * automatic retry (fresh cache-busting query param after
 * PRODUCT_IMAGE_RETRY_DELAY_MS) so a transient CDN hiccup self-heals, then an
 * explicit 'unavailable' phase — the caller renders a labeled placeholder
 * instead of a silent grey tile pretending to be the product photo.
 * A missing/invalid URL skips straight to 'unavailable'.
 */
export function useProductImageRetry(rawUri: string | undefined | null): ProductImageRetryState {
  const uri = useMemo(() => usableProductImageUri(rawUri), [rawUri]);
  const [retryUri, setRetryUri] = useState('');
  const [unavailable, setUnavailable] = useState(!uri);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const failedUriRef = useRef('');

  useEffect(() => {
    // New slot content restarts the bounded retry cycle.
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    failedUriRef.current = '';
    setRetryUri('');
    setUnavailable(!uri);
    return () => {
      if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    };
  }, [uri]);

  const onImageError = useCallback(() => {
    if (!uri || unavailable) return;
    const active = retryUri || uri;
    // Loaders can redeliver an error for a source already being retried;
    // only the first error per displayed URI advances the state machine.
    if (failedUriRef.current === active) return;
    failedUriRef.current = active;
    if (!retryUri && bustableProductImageUri(uri)) {
      // The single automatic retry. A second failure lands on the
      // 'unavailable' branch below — never a third attempt.
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        setRetryUri(cacheBustedImageUri(uri));
      }, PRODUCT_IMAGE_RETRY_DELAY_MS);
    } else {
      setUnavailable(true);
    }
  }, [uri, retryUri, unavailable]);

  return {
    phase: unavailable ? 'unavailable' : 'ready',
    activeUri: retryUri || uri,
    onImageError,
  };
}

/**
 * Explicit 'photo unavailable' tile: muted background + icon glyph + the
 * localized label. Rendered instead of the image — never on top of a
 * blank one — so a failed photo can never pass as a product image.
 */
export function ImageUnavailablePlaceholder({
  label,
  style,
  iconSize = 20,
  testID,
}: {
  label: string;
  style?: StyleProp<ViewStyle>;
  iconSize?: number;
  testID?: string;
}) {
  return (
    <View
      testID={testID}
      style={[styles.base, style]}
      accessibilityRole="image"
      accessibilityLabel={label}
    >
      <Ionicons name="image-outline" size={iconSize} color="#8A97A8" />
      {label ? <ScaledText style={styles.label}>{label}</ScaledText> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#E9EEF5',
    gap: 4,
  },
  label: {
    fontSize: 10,
    fontWeight: '600',
    color: '#6B7A8F',
    textAlign: 'center',
    paddingHorizontal: 4,
  },
});
