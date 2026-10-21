const mockUpload = jest.fn();
const mockPublicUrl = jest.fn((path: string) => ({ data: { publicUrl: `https://storage.test/storage/v1/object/public/avatars/${path}` } }));
jest.mock('../../lib/supabase', () => ({ supabase: { storage: { from: () => ({ upload: mockUpload, getPublicUrl: mockPublicUrl }) } } }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
import { AVATAR_MAX_BYTES, AvatarUploadError, avatarRequestHeaders, uploadProfileAvatar } from '../../src/services/profileAvatar';
const originalFetch = global.fetch;
beforeEach(() => {
  jest.clearAllMocks();
  mockUpload.mockResolvedValue({ error: null });
  global.fetch = jest.fn().mockResolvedValue({ arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer });
});
afterAll(() => { global.fetch = originalFetch; });
test('uploads native image bytes with canonical JPEG MIME and an owner-scoped unique path', async () => {
  const result = await uploadProfileAvatar('owner', 'file:///cache/avatar.JPG?version=1', 'image/jpeg');
  expect(mockUpload.mock.calls[0][0]).toMatch(/^owner\/avatar_.*\.jpg$/);
  expect(mockUpload.mock.calls[0][1]).toBeInstanceOf(ArrayBuffer);
  expect(mockUpload.mock.calls[0][2]).toEqual({ contentType: 'image/jpeg', upsert: false });
  expect(result.url).toBe(`https://storage.test/storage/v1/object/authenticated/avatars/${result.path}`);
});
test('preserves PNG format rather than declaring image/jpg', async () => {
  await uploadProfileAvatar('owner', 'file:///cache/photo.png');
  expect(mockUpload.mock.calls[0][2].contentType).toBe('image/png');
});
test('does not treat an upload error as a saved photo', async () => {
  const failure = new Error('Storage unavailable');
  mockUpload.mockResolvedValue({ error: failure });
  await expect(uploadProfileAvatar('owner', 'file:///photo.jpg')).rejects.toBe(failure);
  expect(mockPublicUrl).not.toHaveBeenCalled();
});
test.each([0, AVATAR_MAX_BYTES + 1])('rejects empty/oversized files before upload (%s bytes)', async size => {
  global.fetch = jest.fn().mockResolvedValue({ arrayBuffer: async () => new ArrayBuffer(size) });
  await expect(uploadProfileAvatar('owner', 'file:///photo.jpg')).rejects.toMatchObject({ code: 'invalid_size' });
  expect(mockUpload).not.toHaveBeenCalled();
});
test('rejects unsupported content types before reading or uploading', async () => {
  await expect(uploadProfileAvatar('owner', 'file:///photo.svg', 'image/svg+xml')).rejects.toBeInstanceOf(AvatarUploadError);
  expect(global.fetch).not.toHaveBeenCalled();
  expect(mockUpload).not.toHaveBeenCalled();
});
test('only attaches credentials to this project’s private-avatar endpoint', () => {
  expect(avatarRequestHeaders('https://storage.test/storage/v1/object/authenticated/avatars/owner/a.png', 'session')).toEqual({ Authorization: 'Bearer session' });
  for (const uri of ['https://external.test/storage/v1/object/authenticated/avatars/a.png', 'https://storage.test.evil.test/storage/v1/object/authenticated/avatars/a.png', 'https://storage.test/storage/v1/object/public/avatars/a.png', 'file:///cache/avatar.png']) {
    expect(avatarRequestHeaders(uri, 'session')).toBeUndefined();
  }
});
