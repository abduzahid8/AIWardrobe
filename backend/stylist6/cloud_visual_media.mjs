/**
 * Isolated Stylist 6 cloud-only image normalizer. Never fetches or uploads
 * anything: the caller must already validate the seller image URL against
 * both HTTP/download and explicit external-inference allowlists.
 *
 * The hosted NVIDIA critic requires bounded, genuine JPEG bytes. Modern
 * retailer CDNs frequently return PNG/WebP and JPEG files over 512 KiB.
 * Previously all such images abstained (image_toolchain_unavailable).
 *
 * Sharp here runs in the Node compute function (never Deno, production
 * Stylist 5, Expo client, Supabase database or a third-party converter).
 * Decode has limits and only static, physically valid source images pass.
 */
import sharp from 'sharp';

export const CLOUD_VISUAL_MAX_INPUT_BYTES = 4 * 1024 * 1024;
export const CLOUD_VISUAL_MAX_OUTPUT_BYTES = 512 * 1024;
export const CLOUD_VISUAL_MAX_INPUT_PIXELS = 16 * 1024 * 1024;
export const CLOUD_VISUAL_MAX_EDGE = 2048;
const FORMATS = new Set(['jpeg', 'png', 'webp', 'avif']);

// A single decode in-memory, at most two size/quality iterations. Do not
// ingest URLs, images from an arbitrary local filesystem or animation.
export async function prepareCloudVisualMedia(bytes, {
  sharpImpl = sharp,
  maxBytes = CLOUD_VISUAL_MAX_OUTPUT_BYTES,
} = {}) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 128 ||
      bytes.length > CLOUD_VISUAL_MAX_INPUT_BYTES ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 16 * 1024 ||
      maxBytes > CLOUD_VISUAL_MAX_OUTPUT_BYTES) return null;
  let source;
  try {
    source = sharpImpl(bytes, {
      failOn: 'error',
      limitInputPixels: CLOUD_VISUAL_MAX_INPUT_PIXELS,
      animated: false,
      sequentialRead: true,
    });
    const m = await source.metadata();
    // sharp reports AVIF as HEIF with av1 codec, while HEIC uses hevc.
    // This is intentional: do not auto-accept every HEIF/HEIC asset.
    if (!(FORMATS.has(m.format) || (m.format === 'heif' && m.compression === 'av1')) ||
        !Number.isSafeInteger(m.width) || !Number.isSafeInteger(m.height) ||
        m.width < 64 || m.height < 64 ||
        m.width > 8192 || m.height > 8192 ||
        m.width * m.height > CLOUD_VISUAL_MAX_INPUT_PIXELS ||
        (m.pages != null && m.pages !== 1)) return null;
    // Do not relay source metadata/EXIF or alpha. JPEG normalization
    // strips metadata, flattens transparency and rotates EXIF orientation.
    for (const [edge, quality] of [[1024, 74], [768, 52]]) {
      const result = await sharpImpl(bytes, {
        failOn: 'error',
        limitInputPixels: CLOUD_VISUAL_MAX_INPUT_PIXELS,
        animated: false,
        sequentialRead: true,
      })
        .rotate()
        .resize(edge, edge, { fit: 'inside', withoutEnlargement: true })
        .flatten({ background: '#ffffff' })
        .jpeg({ quality, progressive: false, mozjpeg: false })
        .toBuffer();
      if (result.length >= 128 && result.length <= maxBytes &&
          result[0] === 0xff && result[1] === 0xd8 &&
          result[result.length - 2] === 0xff && result[result.length - 1] === 0xd9) {
        return result;
      }
    }
  } catch {
    // Fail closed, never substitute different product photos or fabricate
    // pixels. A caller may explain an abstention, not silently claim JUDGED.
  }
  return null;
}

/**
 * Compare source/target format only for aggregate monitoring. This never
 * exposes seller URL, image bytes, account identifiers or EXIF.
 */
export function cloudVisualMediaClass(bytes) {
  if (!Buffer.isBuffer(bytes) || !bytes.length) return 'missing';
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8) return 'jpeg';
  if (bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) return 'png';
  if (bytes.length >= 12 && bytes.toString('ascii',0,4) === 'RIFF' &&
      bytes.toString('ascii',8,12) === 'WEBP') return 'webp';
  if (bytes.length >= 12 && bytes.toString('ascii',4,8) === 'ftyp') return 'heif_avif';
  return 'other';
}
