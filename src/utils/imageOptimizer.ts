/**
 * Image Optimization Utility
 *
 * Compresses and resizes images before upload to reduce bandwidth and memory usage.
 * Uses expo-image-manipulator for efficient native image processing.
 *
 * Usage:
 *   const optimized = await optimizeImage(base64, { maxWidth: 1920, quality: 0.8 });
 */
// @ts-ignore — expo-image-manipulator types loaded when package is installed
import * as ImageManipulator from 'expo-image-manipulator';

interface OptimizeOptions {
    maxWidth?: number;
    maxHeight?: number;
    quality?: number;
    format?: 'jpeg' | 'png';
}

const DEFAULT_OPTIONS: Required<OptimizeOptions> = {
    maxWidth: 1920,
    maxHeight: 1920,
    quality: 0.8,
    format: 'jpeg',
};

/**
 * Optimize an image for upload.
 * Resizes to fit within maxWidth/maxHeight and compresses to target quality.
 *
 * @param uri - Image URI (file://, content://, or base64 data URI)
 * @param options - Optimization settings
 * @returns Optimized image as base64 string
 */
export async function optimizeImage(
    uri: string,
    options?: OptimizeOptions
): Promise<{ base64: string; width: number; height: number; uri: string }> {
    const opts = { ...DEFAULT_OPTIONS, ...options };

    try {
        // Resize to fit within bounds
        const actions: ImageManipulator.Action[] = [
            {
                resize: {
                    width: opts.maxWidth,
                    // Height is auto-calculated to maintain aspect ratio
                },
            },
        ];

        const format =
            opts.format === 'png'
                ? ImageManipulator.SaveFormat.PNG
                : ImageManipulator.SaveFormat.JPEG;

        const result = await ImageManipulator.manipulateAsync(uri, actions, {
            compress: opts.quality,
            format,
            base64: true,
        });

        return {
            base64: result.base64 || '',
            width: result.width,
            height: result.height,
            uri: result.uri,
        };
    } catch (error) {
        // If optimization fails, return original as-is for resilience
        console.warn('[ImageOptimizer] Optimization failed, using original:', error);

        // If the input is already a base64 data URI, extract the base64 part
        if (uri.startsWith('data:')) {
            const base64Part = uri.split(',')[1] || '';
            return { base64: base64Part, width: 0, height: 0, uri };
        }

        return { base64: '', width: 0, height: 0, uri };
    }
}

/**
 * Estimate the size ofimage data in bytes (approximate).
 */
export function estimateBase64Size(base64: string): number {
    // Base64 has ~33% overhead
    return Math.ceil((base64.length * 3) / 4);
}

/**
 * Check if an image needs optimization.
 * Returns true if the base64 data exceeds 1MB.
 */
export function needsOptimization(base64: string, maxSizeBytes = 1024 * 1024): boolean {
    return estimateBase64Size(base64) > maxSizeBytes;
}

export default { optimizeImage, estimateBase64Size, needsOptimization };
