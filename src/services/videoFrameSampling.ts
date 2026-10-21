import { createVideoPlayer, type VideoPlayer } from 'expo-video';
import * as FileSystem from 'expo-file-system/legacy';

/** Sample the beginning, middle and end without requesting the end-of-file boundary. */
export function videoSampleTimes(durationSeconds: number): number[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) throw new Error('Video duration unavailable');
  const durationMs = durationSeconds * 1000;
  if (!Number.isSafeInteger(Math.ceil(durationMs))) throw new Error('Invalid video duration');
  return [...new Set([0.1, 0.5, 0.9].map(fraction => Math.floor(durationMs * fraction)))];
}

/** Read the actual file on each attempt, including a rebased offline queue file. */
async function withLoadedVideo<T>(uri: string, signal: AbortSignal | undefined, consume: (player: VideoPlayer, duration: number) => Promise<T>): Promise<T> {
  const aborted = () => Object.assign(new Error('Video analysis cancelled'), { name: 'AbortError' });
  if (signal?.aborted) throw aborted();
  // An empty player plus replaceAsync avoids synchronous asset loading on iOS.
  const player = createVideoPlayer(null);
  const subscriptions: Array<{ remove(): void }> = [];
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  try {
    const duration = await new Promise<number>((resolve, reject) => {
      let settled = false;
      const finish = (value?: number, error?: Error) => {
        if (settled) return;
        settled = true;
        if (error) reject(error);
        else {
          try { videoSampleTimes(value!); resolve(value!); }
          catch (validationError) { reject(validationError); }
        }
      };
      subscriptions.push(player.addListener('sourceLoad', event => {
        // Clearing the empty source can emit its own metadata event.
        if (event.videoSource != null) finish(event.duration);
      }));
      subscriptions.push(player.addListener('statusChange', event => {
        if (event.status === 'error') finish(undefined, new Error('Unable to read video metadata'));
      }));
      timer = setTimeout(() => finish(undefined, new Error('Video metadata timed out')), 10000);
      onAbort = () => finish(undefined, aborted());
      signal?.addEventListener('abort', onAbort, { once: true });
      if (signal?.aborted) { onAbort(); return; }
      void player.replaceAsync(uri).then(() => {
        // Loading may finish after cancellation/sourceLoad already released it.
        if (settled) return;
        try {
          const loadedDuration = player.duration;
          if (loadedDuration > 0) finish(loadedDuration);
        } catch { finish(undefined, new Error('Unable to read video duration')); }
      }, () => finish(undefined, new Error('Unable to load video')));
    });
    return await consume(player, duration);
  } finally {
    if (timer) clearTimeout(timer);
    if (onAbort) signal?.removeEventListener('abort', onAbort);
    for (const subscription of subscriptions) {
      try { subscription.remove(); } catch { /* Release the other listeners and player too. */ }
    }
    player.release();
  }
}

export function readVideoSampleTimes(uri: string, signal?: AbortSignal): Promise<number[]> {
  return withLoadedVideo(uri, signal, async (_player, duration) => videoSampleTimes(duration));
}

/** Android's deprecated thumbnail module selects keyframes; ExpoVideo selects closest frames. */
export function readAccurateVideoFrames(uri: string, signal?: AbortSignal): Promise<Array<{ timeMs: number; base64: string }>> {
  return withLoadedVideo(uri, signal, async (player, duration) => {
    const { ImageManipulator, SaveFormat } = require('expo-image-manipulator') as typeof import('expo-image-manipulator');
    const times = videoSampleTimes(duration);
    const thumbnails = await player.generateThumbnailsAsync(times.map(time => time / 1000), { maxWidth: 1600, maxHeight: 1600 });
    try {
      if (thumbnails.length !== times.length) throw new Error('Incomplete video frames');
      const frames = [];
      for (let index = 0; index < thumbnails.length; index++) {
        if (signal?.aborted) throw Object.assign(new Error('Video analysis cancelled'), { name: 'AbortError' });
        const context = ImageManipulator.manipulate(thumbnails[index]);
        let image: Awaited<ReturnType<typeof context.renderAsync>> | undefined;
        let temporary: string | undefined;
        try {
          image = await context.renderAsync();
          const saved = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.9, base64: true });
          temporary = saved.uri;
          if (!saved.base64) throw new Error('Video frame encoding failed');
          frames.push({ timeMs: times[index], base64: saved.base64 });
        } finally {
          image?.release(); context.release();
          if (temporary) await FileSystem.deleteAsync(temporary, { idempotent: true });
        }
      }
      if (signal?.aborted) throw Object.assign(new Error('Video analysis cancelled'), { name: 'AbortError' });
      return frames;
    } finally {
      for (const thumbnail of thumbnails) thumbnail.release();
    }
  });
}
