import * as FileSystem from 'expo-file-system/legacy';
import { ImageManipulator } from 'expo-image-manipulator';
import { createVideoPlayer } from 'expo-video';
import { readVideoSampleTimes, videoSampleTimes, readAccurateVideoFrames } from '../../src/services/videoFrameSampling';
jest.mock('expo-video', () => ({ createVideoPlayer: jest.fn() }));
jest.mock('expo-file-system/legacy',()=>({deleteAsync:jest.fn().mockResolvedValue(undefined)}));
jest.mock('expo-image-manipulator',()=>({ImageManipulator:{manipulate:jest.fn()},SaveFormat:{JPEG:'jpeg'}}));
function fixture() {
  const events: Record<string, (value: any) => void> = {};
  const removals: jest.Mock[] = [];
  const player = {
    duration: 0,
    generateThumbnailsAsync: jest.fn(),
    addListener: jest.fn((name: string, listener: (value: any) => void) => {
      events[name] = listener; const remove = jest.fn(); removals.push(remove); return { remove };
    }),
    replaceAsync: jest.fn().mockResolvedValue(undefined), release: jest.fn(),
  };
  (createVideoPlayer as jest.Mock).mockReturnValueOnce(player);
  return { player, events, removals };
}
beforeEach(() => { jest.clearAllMocks(); });
afterEach(() => { jest.useRealTimers(); });
it.each([
  [30, [3000, 15000, 27000]], [120, [12000, 60000, 108000]],
  [1, [100, 500, 900]], [0.001, [0]],
])('samples across a %ss clip rather than just its first two seconds', (duration, expected) => {
  expect(videoSampleTimes(duration as number)).toEqual(expected);
});
it.each([0, -1, NaN, Infinity])('rejects unavailable duration %s', duration => {
  expect(() => videoSampleTimes(duration)).toThrow();
});
it('loads metadata asynchronously and always releases the temporary player and listeners', async () => {
  const { player, events, removals } = fixture();
  const pending = readVideoSampleTimes('file:///managed/clip.mp4');
  expect(createVideoPlayer).toHaveBeenCalledWith(null);
  expect(player.replaceAsync).toHaveBeenCalledWith('file:///managed/clip.mp4');
  events.sourceLoad({ videoSource: { uri:'file:///fixture.mp4' }, duration: 30 });
  expect(await pending).toEqual([3000,15000,27000]);
  expect(player.release).toHaveBeenCalledTimes(1);
  removals.forEach(remove => expect(remove).toHaveBeenCalledTimes(1));
});
it('a restored queue URI is reread instead of using duration from an old container', async () => {
  const first = fixture(); const firstRead = readVideoSampleTimes('file:///old/clip.mp4');
  first.events.sourceLoad({ videoSource: { uri:'file:///fixture.mp4' }, duration:10 }); await firstRead;
  const restored = fixture(); const nextRead = readVideoSampleTimes('file:///new/clip.mp4');
  restored.events.sourceLoad({ videoSource: { uri:'file:///fixture.mp4' }, duration:60 });
  expect(await nextRead).toEqual([6000,30000,54000]);
  expect(restored.player.replaceAsync).toHaveBeenCalledWith('file:///new/clip.mp4');
});
it.each(['load', 'status', 'invalid', 'abort', 'timeout'])('cleans up after metadata %s failure', async mode => {
  jest.useFakeTimers();
  const { player, events, removals } = fixture();
  if (mode === 'load') player.replaceAsync.mockRejectedValueOnce(new Error('native failure'));
  const controller = new AbortController();
  const pending = readVideoSampleTimes('file:///bad.mp4', controller.signal);
  const rejected = expect(pending).rejects.toThrow();
  if (mode === 'status') events.statusChange({ status:'error' });
  if (mode === 'invalid') events.sourceLoad({ videoSource: { uri:'file:///fixture.mp4' }, duration:0 });
  if (mode === 'abort') controller.abort();
  if (mode === 'timeout') await jest.advanceTimersByTimeAsync(10000);
  await rejected;
  expect(player.release).toHaveBeenCalledTimes(1);
  removals.forEach(remove => expect(remove).toHaveBeenCalledTimes(1));
  expect(jest.getTimerCount()).toBe(0);
});
it('an already cancelled request does not create a player', async () => {
  const controller = new AbortController(); controller.abort();
  await expect(readVideoSampleTimes('file:///clip.mp4', controller.signal)).rejects.toMatchObject({ name:'AbortError' });
  expect(createVideoPlayer).not.toHaveBeenCalled();
});

it('ignores metadata emitted for the initial empty source', async () => {
  const { events } = fixture(); const pending = readVideoSampleTimes('file:///clip.mp4');
  events.sourceLoad({ videoSource:null, duration:0 });
  events.sourceLoad({ videoSource:{uri:'file:///clip.mp4'}, duration:10 });
  expect(await pending).toEqual([1000,5000,9000]);
});

it('late native load completion cannot access an already released player', async () => {
  const { player } = fixture();
  let complete!: () => void;
  player.replaceAsync.mockReturnValueOnce(new Promise<void>(resolve => { complete = resolve; }));
  const readDuration = jest.fn(() => { throw new Error('released native object'); });
  Object.defineProperty(player, 'duration', { get:readDuration });
  const controller = new AbortController();
  const pending = readVideoSampleTimes('file:///clip.mp4', controller.signal);
  controller.abort(); await expect(pending).rejects.toThrow();
  complete(); await Promise.resolve();
  expect(readDuration).not.toHaveBeenCalled();
});

it('the exact-frame API uses seconds and encodes each native image while cleaning resources', async () => {
  const {player,events}=fixture();
  const thumbnails=[0,1,2].map(()=>({release:jest.fn()}));
  player.generateThumbnailsAsync.mockResolvedValue(thumbnails);
  const image={saveAsync:jest.fn().mockResolvedValue({uri:'file:///encoded.jpg',base64:'encoded'}),release:jest.fn()};
  const context={renderAsync:jest.fn().mockResolvedValue(image),release:jest.fn()};
  (ImageManipulator.manipulate as jest.Mock).mockReturnValue(context);
  const pending=readAccurateVideoFrames('file:///clip.mp4');events.sourceLoad({videoSource:{uri:'file:///clip.mp4'},duration:30});
  expect(await pending).toEqual([3000,15000,27000].map(timeMs=>({timeMs,base64:'encoded'})));
  expect(player.generateThumbnailsAsync).toHaveBeenCalledWith([3,15,27],{maxWidth:1600,maxHeight:1600});
  expect(FileSystem.deleteAsync).toHaveBeenCalledTimes(3);expect(image.release).toHaveBeenCalledTimes(3);expect(context.release).toHaveBeenCalledTimes(3);
  thumbnails.forEach(thumbnail=>expect(thumbnail.release).toHaveBeenCalledTimes(1));expect(player.release).toHaveBeenCalledTimes(1);
});
it('encoding failure retains no native image references or partial returned batch', async () => {
  const {player,events}=fixture();const thumbnails=[0,1,2].map(()=>({release:jest.fn()}));
  player.generateThumbnailsAsync.mockResolvedValue(thumbnails);
  const context={renderAsync:jest.fn().mockRejectedValue(new Error('encode failed')),release:jest.fn()};
  (ImageManipulator.manipulate as jest.Mock).mockReturnValue(context);
  const pending=readAccurateVideoFrames('file:///clip.mp4');events.sourceLoad({videoSource:{uri:'file:///clip.mp4'},duration:30});
  await expect(pending).rejects.toThrow('encode failed');expect(context.release).toHaveBeenCalledTimes(1);
  thumbnails.forEach(thumbnail=>expect(thumbnail.release).toHaveBeenCalledTimes(1));expect(player.release).toHaveBeenCalledTimes(1);
});
