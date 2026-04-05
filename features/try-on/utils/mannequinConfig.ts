/**
 * Mannequin Model Configuration
 *
 * For App Store distribution, the 3D model is hosted on Supabase Storage
 * and loaded remotely — keeping the app bundle small.
 *
 * HOW TO UPLOAD YOUR MODEL:
 * 1. Export your .blend file from Blender:
 *    File → Export → glTF 2.0 (.glb)
 *    Enable Draco Compression for smaller file size
 * 2. In Supabase Dashboard → Storage → Create bucket "models" (public)
 * 3. Upload your .glb file
 * 4. Copy the public URL and paste it below
 */

// ---------------------------------------------------------------------------
// REMOTE URL (primary — App Store friendly, no bundle size impact)
// Replace with your Supabase Storage public URL after upload.
// ---------------------------------------------------------------------------
export const MANNEQUIN_MODEL_URL =
  'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/models/mannequin_male.glb';

// ---------------------------------------------------------------------------
// FALLBACK FLAG
// When true, falls back to the procedural Three.js mannequin if the remote
// model fails to load (network error, model not uploaded yet, etc.)
// ---------------------------------------------------------------------------
export const MANNEQUIN_USE_PROCEDURAL_FALLBACK = true;
