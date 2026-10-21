import AsyncStorage from '@react-native-async-storage/async-storage';
import { uploadQueue } from '../../src/services/uploadQueue';

jest.mock('@react-native-community/netinfo', () => ({ __esModule: true, default: { fetch: jest.fn() } }));

it('reports failed disk removal while hiding the previous account queue from listeners', async () => {
    const failure = new Error('synthetic removal failure');
    jest.mocked(AsyncStorage.removeItem).mockRejectedValueOnce(failure);
    const listener = jest.fn();
    const stop = uploadQueue.subscribe(listener);
    try {
        await expect(uploadQueue.clearQueue()).rejects.toBe(failure);
        expect(listener).toHaveBeenCalledWith(expect.objectContaining({ queue: [], pendingCount: 0 }));
    } finally { stop(); }
});
