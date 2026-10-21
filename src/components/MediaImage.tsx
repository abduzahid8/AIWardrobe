import React, { forwardRef, useMemo, useSyncExternalStore, useRef } from 'react';
import { Image as NativeImage, ImageProps as NativeImageProps, Platform } from 'react-native';
import { Image as BaseExpoImage, ImageProps as ExpoImageProps } from 'expo-image';
import useAuthStore from '../../store/auth';
import { authEvents } from '../../store/authEvents';
import { wardrobeMediaReference } from '../services/wardrobeMedia';
import { nextMediaImageLifecycle, type MediaImageLifecycle } from '../utils/mediaImageLifecycle';
import { ImageIdentity, mediaImageSource, nativeMediaImageSource, prefetchMediaImages } from '../utils/mediaImageSource';

const subscribeIdentity = (notify: () => void) => {
  const login = authEvents.onLogin(() => notify()), logout = authEvents.onLogout(notify);
  return () => { login(); logout(); };
};
function currentIdentity(): ImageIdentity {
  const session = useAuthStore.getState().session;
  return session && authEvents.canApplyToUser(session.user.id)
    ? { userId: session.user.id, accessToken: session.access_token } : {};
}
function useImageIdentity(): ImageIdentity {
  useAuthStore(state => state.session);
  useSyncExternalStore(subscribeIdentity, authEvents.getRevision, authEvents.getRevision);
  return currentIdentity();
}

function useMediaViewKey(source: unknown, identity: ImageIdentity, recyclingKey?: string | null): string {
  const lifecycle = useRef<MediaImageLifecycle | undefined>(undefined);
  lifecycle.current = nextMediaImageLifecycle(lifecycle.current, source, identity, recyclingKey);
  return String(lifecycle.current.version);
}

const Native = forwardRef<React.ElementRef<typeof NativeImage>, NativeImageProps>((props, ref) => {
  const { userId, accessToken } = useImageIdentity();
  const identity = useMemo(() => ({ userId, accessToken }), [userId, accessToken]);
  const source = useMemo(() => mediaImageSource(props.source, identity), [props.source, identity]);
  const nativeSource = useMemo(() => nativeMediaImageSource(source, Platform.OS, {width: props.width, height: props.height}), [source, props.width, props.height]);
  const viewKey = useMediaViewKey(nativeSource, identity);
  return <NativeImage {...props} key={viewKey} ref={ref} source={nativeSource as NativeImageProps['source']}
    defaultSource={mediaImageSource(props.defaultSource, identity) as NativeImageProps['defaultSource']}
    loadingIndicatorSource={mediaImageSource(props.loadingIndicatorSource, identity) as NativeImageProps['loadingIndicatorSource']} />;
});
Native.displayName = 'MediaImage';
Object.setPrototypeOf(Native, NativeImage);
export const Image = Object.assign(Native, {
  prefetch: ((uri: string) => wardrobeMediaReference(uri) ? Promise.resolve(false) : NativeImage.prefetch(uri)) as typeof NativeImage.prefetch,
  getSize: ((uri: string, success?: (width: number, height: number) => void, failure?: (error: unknown) => void) => {
    if (!wardrobeMediaReference(uri)) return success ? NativeImage.getSize(uri, success, failure) : NativeImage.getSize(uri);
    const source = mediaImageSource(uri, currentIdentity()) as { uri: string; headers: Record<string, string> } | undefined;
    if (!source) {
      const error = new Error('Private image unavailable');
      if (success) { failure?.(error); return; }
      return Promise.reject(error);
    }
    return success ? NativeImage.getSizeWithHeaders(source.uri, source.headers, success, failure)
      : NativeImage.getSizeWithHeaders(source.uri, source.headers);
  }) as typeof NativeImage.getSize,
}) as unknown as typeof NativeImage;

const Expo = forwardRef<React.ElementRef<typeof BaseExpoImage>, ExpoImageProps>((props, ref) => {
  const { userId, accessToken } = useImageIdentity();
  const identity = useMemo(() => ({ userId, accessToken }), [userId, accessToken]);
  const source = useMemo(() => mediaImageSource(props.source, identity), [props.source, identity]);
  const viewKey = useMediaViewKey(source, identity, props.recyclingKey);
  return <BaseExpoImage {...props} key={viewKey} ref={ref} source={source as ExpoImageProps['source']}
    placeholder={mediaImageSource(props.placeholder, identity) as ExpoImageProps['placeholder']} />;
});
Expo.displayName = 'MediaExpoImage';
Object.setPrototypeOf(Expo, BaseExpoImage);
export const ExpoImage = Object.assign(Expo, {
  prefetch: ((urls, options) => prefetchMediaImages(urls, options, currentIdentity(),
    (uri, settings) => typeof settings === 'string' ? BaseExpoImage.prefetch(uri, settings)
      : BaseExpoImage.prefetch(uri, settings))) as typeof BaseExpoImage.prefetch,
}) as unknown as typeof BaseExpoImage;
