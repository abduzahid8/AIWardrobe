/**
 * POST   /body-profiles          — create a body profile
 * GET    /body-profiles/me       — fetch the active profile for the current user
 * PATCH  /body-profiles/:id      — partial update (height, weight, body type, measurements, privacy)
 * DELETE /body-profiles/:id      — delete a profile
 *
 * Persistence: Supabase table `body_profiles` (see supabase/migrations/022_body_profiles.sql).
 * On PATCH the engine version is bumped — see bodyProfileStore.ts on the mobile side.
 *
 * All routes require a valid Supabase JWT (authenticateToken). Validation rules
 * are in api/middleware/validators.js.
 */

import express from 'express';
import { supabase } from '../lib/supabase.js';
import { authenticateToken } from '../middleware/auth.js';
import { handleValidationErrors } from '../middleware/validators.js';
import { body, param } from 'express-validator';
import logger from '../utils/logger.js';

const router = express.Router();

// ─── Helpers ─────────────────────────────────────────────────────────────────

const BODY_TYPES = ['ectomorph', 'average', 'mesomorph', 'endomorph', 'hourglass', 'pear'];
const GENDERS = ['male', 'female', 'other', 'prefer_not_to_say'];
const SOURCES = ['manual', 'apple_measure', 'photo_sam_3d_body', 'arkit_height', 'hybrid'];
const CONFIDENCES = ['low', 'medium', 'high'];
const STATUSES = ['draft', 'analyzing', 'ready', 'failed'];

const MEASUREMENT_ZONES = [
  'shoulderWidth', 'chest', 'waist', 'hips',
  'torsoLength', 'armLength', 'sleeveLength',
  'inseam', 'thigh', 'calf', 'footLength',
];

function rowToProfile(row) {
  if (!row) return null;
  return {
    id: row.id,
    userId: row.user_id,
    name: row.name || undefined,
    status: row.status,
    isActive: row.is_active,
    gender: row.gender || undefined,
    height: row.height_value_cm != null ? {
      valueCm: row.height_value_cm,
      confidence: row.height_confidence,
      source: row.height_source,
      updatedAt: row.height_updated_at,
    } : null,
    weightKg: row.weight_kg,
    bodyType: row.body_type,
    measurements: row.measurements || {},
    mesh: row.mesh || undefined,
    privacy: row.privacy || { retainSourcePhoto: false, retainMesh: true },
    version: row.version,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

// ─── Validators ──────────────────────────────────────────────────────────────

const validateCreate = [
  body('height.valueCm')
    .isFloat({ min: 80, max: 250 })
    .withMessage('height.valueCm must be 80-250 cm'),
  body('height.source')
    .optional()
    .isIn(SOURCES)
    .withMessage('invalid height.source'),
  body('weightKg')
    .optional()
    .isFloat({ min: 20, max: 300 })
    .withMessage('weightKg must be 20-300 kg'),
  body('bodyType')
    .optional()
    .isIn(BODY_TYPES)
    .withMessage('invalid bodyType'),
  body('gender')
    .optional()
    .isIn(GENDERS)
    .withMessage('invalid gender'),
  body('measurements')
    .optional()
    .isObject()
    .withMessage('measurements must be an object'),
  body('privacy.retainSourcePhoto')
    .optional()
    .isBoolean(),
  body('privacy.retainMesh')
    .optional()
    .isBoolean(),
  handleValidationErrors,
];

const validatePatch = [
  param('id').isUUID().withMessage('id must be a UUID'),
  body('height.valueCm').optional().isFloat({ min: 80, max: 250 }),
  body('height.source').optional().isIn(SOURCES),
  body('height.confidence').optional().isIn(CONFIDENCES),
  body('weightKg').optional().isFloat({ min: 20, max: 300 }),
  body('bodyType').optional().isIn(BODY_TYPES),
  body('gender').optional().isIn(GENDERS),
  body('status').optional().isIn(STATUSES),
  body('measurements').optional().isObject(),
  body('measurements.*.valueCm').optional().isFloat({ min: 0, max: 300 }),
  body('measurements.*.confidence').optional().isIn(CONFIDENCES),
  body('measurements.*.source').optional().isIn(SOURCES),
  handleValidationErrors,
];

const validateId = [
  param('id').isUUID().withMessage('id must be a UUID'),
  handleValidationErrors,
];

// ─── Routes ──────────────────────────────────────────────────────────────────

/**
 * POST /body-profiles
 * Create a new body profile for the authenticated user.
 * If `isActive` is true (or omitted) and no other active profile exists,
 * this profile becomes active.
 */
router.post('/', authenticateToken, validateCreate, async (req, res) => {
  try {
    const userId = req.user.id;
    const { height, weightKg, bodyType, gender, measurements, privacy, name } = req.body;

    // If a profile will be active, deactivate any existing active ones.
    const wantsActive = req.body.isActive !== false;

    if (wantsActive) {
      await supabase
        .from('body_profiles')
        .update({ is_active: false })
        .eq('user_id', userId)
        .eq('is_active', true);
    }

    const insertRow = {
      user_id: userId,
      name: name || null,
      status: 'draft',
      is_active: wantsActive,
      gender: gender || null,
      height_value_cm: height?.valueCm ?? null,
      height_confidence: height?.confidence || 'medium',
      height_source: height?.source || 'manual',
      weight_kg: weightKg ?? null,
      body_type: bodyType || null,
      measurements: measurements || {},
      privacy: privacy || { retainSourcePhoto: false, retainMesh: true },
      version: 1,
    };

    const { data, error } = await supabase
      .from('body_profiles')
      .insert([insertRow])
      .select()
      .single();

    if (error) throw error;
    logger.info(`✅ Created body profile ${data.id} for user ${userId}`);
    res.status(201).json({ success: true, profile: rowToProfile(data) });
  } catch (err) {
    logger.error('POST /body-profiles failed:', err.message);
    res.status(500).json({ error: 'Failed to create body profile' });
  }
});

/**
 * GET /body-profiles/me
 * Returns the user's active body profile. Falls back to the most recent
 * profile if none is marked active.
 */
router.get('/me', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;

    // Active first
    const { data: active, error: activeErr } = await supabase
      .from('body_profiles')
      .select('*')
      .eq('user_id', userId)
      .eq('is_active', true)
      .maybeSingle();
    if (activeErr) throw activeErr;
    if (active) return res.json({ profile: rowToProfile(active) });

    // Fallback: most recent
    const { data: recent, error: recentErr } = await supabase
      .from('body_profiles')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false })
      .limit(1)
      .maybeSingle();
    if (recentErr) throw recentErr;

    return res.json({ profile: recent ? rowToProfile(recent) : null });
  } catch (err) {
    logger.error('GET /body-profiles/me failed:', err.message);
    res.status(500).json({ error: 'Failed to fetch body profile' });
  }
});

/**
 * GET /body-profiles
 * List all profiles for the current user (for the profile switcher UI).
 */
router.get('/', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { data, error } = await supabase
      .from('body_profiles')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });
    if (error) throw error;
    res.json({ profiles: (data || []).map(rowToProfile) });
  } catch (err) {
    logger.error('GET /body-profiles failed:', err.message);
    res.status(500).json({ error: 'Failed to list body profiles' });
  }
});

/**
 * PATCH /body-profiles/:id
 * Partial update. Bumps `version` + `updated_at` on every write.
 */
router.patch('/:id', authenticateToken, validatePatch, async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;
    const patch = req.body;

    // Verify ownership
    const { data: existing, error: fetchErr } = await supabase
      .from('body_profiles')
      .select('*')
      .eq('id', id)
      .eq('user_id', userId)
      .maybeSingle();
    if (fetchErr) throw fetchErr;
    if (!existing) return res.status(404).json({ error: 'Body profile not found' });

    // Build the column-update object
    const updates = {};
    if (patch.name !== undefined) updates.name = patch.name;
    if (patch.gender !== undefined) updates.gender = patch.gender;
    if (patch.weightKg !== undefined) updates.weight_kg = patch.weightKg;
    if (patch.bodyType !== undefined) updates.body_type = patch.bodyType;
    if (patch.status !== undefined) updates.status = patch.status;
    if (patch.privacy) updates.privacy = patch.privacy;

    if (patch.height) {
      if (patch.height.valueCm !== undefined) updates.height_value_cm = patch.height.valueCm;
      if (patch.height.confidence) updates.height_confidence = patch.height.confidence;
      if (patch.height.source) updates.height_source = patch.height.source;
      updates.height_updated_at = new Date().toISOString();
    }

    if (patch.measurements) {
      // Merge with existing measurements to keep zones not in the patch.
      updates.measurements = { ...(existing.measurements || {}), ...patch.measurements };
    }

    // Bump version
    updates.version = (existing.version || 1) + 1;

    const { data, error } = await supabase
      .from('body_profiles')
      .update(updates)
      .eq('id', id)
      .eq('user_id', userId)
      .select()
      .single();
    if (error) throw error;

    res.json({ success: true, profile: rowToProfile(data) });
  } catch (err) {
    logger.error('PATCH /body-profiles/:id failed:', err.message);
    res.status(500).json({ error: 'Failed to update body profile' });
  }
});

/**
 * POST /body-profiles/:id/activate
 * Mark a profile as the user's active one (deactivates the rest).
 */
router.post('/:id/activate', authenticateToken, validateId, async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;

    // Deactivate all, then activate target. Done in two steps because supabase
    // .eq('user_id', x).eq('is_active', true) is clearer than a sub-select.
    const { data: target, error: tErr } = await supabase
      .from('body_profiles')
      .select('id')
      .eq('id', id)
      .eq('user_id', userId)
      .maybeSingle();
    if (tErr) throw tErr;
    if (!target) return res.status(404).json({ error: 'Body profile not found' });

    await supabase
      .from('body_profiles')
      .update({ is_active: false })
      .eq('user_id', userId);

    const { data, error } = await supabase
      .from('body_profiles')
      .update({ is_active: true })
      .eq('id', id)
      .select()
      .single();
    if (error) throw error;

    res.json({ success: true, profile: rowToProfile(data) });
  } catch (err) {
    logger.error('POST /body-profiles/:id/activate failed:', err.message);
    res.status(500).json({ error: 'Failed to activate body profile' });
  }
});

/**
 * DELETE /body-profiles/:id
 * Permanently delete a body profile and all its measurement history.
 */
router.delete('/:id', authenticateToken, validateId, async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;
    const { error } = await supabase
      .from('body_profiles')
      .delete()
      .eq('id', id)
      .eq('user_id', userId);
    if (error) throw error;
    res.json({ success: true, id });
  } catch (err) {
    logger.error('DELETE /body-profiles/:id failed:', err.message);
    res.status(500).json({ error: 'Failed to delete body profile' });
  }
});

export default router;
