/**
 * POST /api/fit/assess
 *
 * Wraps the fit engine (src/lib/fit/fitEngine.ts) behind an HTTP endpoint.
 * The engine itself is pure, so this is mostly: validate, load body + garment
 * profile, run engine, return FitAssessment JSON.
 *
 * For Month 1, the garment physical profile is loaded from the in-memory
 * seed set in src/types/garment.ts. Month 5 will replace this with a real
 * catalog lookup.
 *
 * This route is mounted at /api/fit and is public (no authenticateToken)
 * because the assessment is a deterministic computation. We do still
 * apply the global apiLimiter.
 */

import express from 'express';
import { supabase } from '../lib/supabase.js';
import { authenticateToken } from '../middleware/auth.js';
import { body } from 'express-validator';
import { handleValidationErrors } from '../middleware/validators.js';
import logger from '../utils/logger.js';

const router = express.Router();

// Importing the engine at runtime requires a TS loader. We re-implement
// the *server-side* apply using a JS-friendly copy of the rules. The
// mobile side uses the full TS engine from src/lib/fit/fitEngine.ts.
//
// This intentionally duplicates the engine logic (rather than wiring TS
// into the Express process) because the api/ folder is plain ESM JS.
// When the engine graduates to "must match" in golden tests, this server
// copy will be replaced by a shared JS build of the same module.

import { assessFit as assessFitCore } from '../services/fitEngine.js';
import { SEED_GARMENT_PHYSICAL_PROFILES, findSeedProfile } from '../services/garmentSeed.js';

const validateAssess = [
  body('bodyProfileId').optional().isUUID(),
  body('garmentId').isString().notEmpty().withMessage('garmentId is required'),
  body('sizeLabel').isString().notEmpty().withMessage('sizeLabel is required'),
  body('bodyProfile').optional().isObject(),
  handleValidationErrors,
];

router.post('/assess', authenticateToken, validateAssess, async (req, res) => {
  try {
    const userId = req.user.id;
    const { bodyProfileId, garmentId, sizeLabel, bodyProfile: inlineBody } = req.body;

    // ── Resolve body profile ────────────────────────────────────────────
    let bodyProfile = inlineBody;
    if (!bodyProfile && bodyProfileId) {
      const { data, error } = await supabase
        .from('body_profiles')
        .select('*')
        .eq('id', bodyProfileId)
        .eq('user_id', userId)
        .maybeSingle();
      if (error) throw error;
      if (!data) return res.status(404).json({ error: 'Body profile not found' });
      bodyProfile = rowToBodyProfile(data);
    }
    if (!bodyProfile) {
      return res.status(400).json({ error: 'bodyProfileId or bodyProfile is required' });
    }

    // ── Resolve garment physical profile ────────────────────────────────
    const garment = findSeedProfile(garmentId, sizeLabel)
      || SEED_GARMENT_PHYSICAL_PROFILES.find(
          (p) => p.garmentId === garmentId && p.sizeLabel === sizeLabel,
        );
    if (!garment) {
      return res.status(404).json({
        error: 'Garment size profile not found',
        hint: 'Seed data only covers a handful of test garments; real catalog lookup comes in Month 5',
      });
    }

    // ── Run engine ──────────────────────────────────────────────────────
    const assessment = assessFitCore(bodyProfile, garment);
    res.json({ success: true, assessment });
  } catch (err) {
    logger.error('POST /api/fit/assess failed:', err.message);
    res.status(500).json({ error: 'Failed to assess fit' });
  }
});

/**
 * POST /api/fit/recommend
 * Compare the current selected size against alternatives and recommend the
 * best-fitting one. Alternatives are looked up from the same seed catalog.
 */
router.post('/recommend', authenticateToken, validateAssess, async (req, res) => {
  try {
    const userId = req.user.id;
    const { bodyProfileId, garmentId, sizeLabel, bodyProfile: inlineBody, alternatives } = req.body;

    let bodyProfile = inlineBody;
    if (!bodyProfile && bodyProfileId) {
      const { data, error } = await supabase
        .from('body_profiles')
        .select('*')
        .eq('id', bodyProfileId)
        .eq('user_id', userId)
        .maybeSingle();
      if (error) throw error;
      if (!data) return res.status(404).json({ error: 'Body profile not found' });
      bodyProfile = rowToBodyProfile(data);
    }
    if (!bodyProfile) {
      return res.status(400).json({ error: 'bodyProfileId or bodyProfile is required' });
    }

    const current = findSeedProfile(garmentId, sizeLabel);
    if (!current) return res.status(404).json({ error: 'Current garment size not found' });

    // Resolve alternatives from the seed catalog. Caller can pass explicit
    // size labels via `alternatives: string[]` or we use the seed set.
    const altSizes = Array.isArray(alternatives) && alternatives.length > 0
      ? alternatives
      : SEED_GARMENT_PHYSICAL_PROFILES
          .filter((p) => p.garmentId === garmentId)
          .map((p) => p.sizeLabel);
    const altProfiles = altSizes
      .filter((s) => s !== sizeLabel)
      .map((s) => findSeedProfile(garmentId, s))
      .filter(Boolean);

    const currentAssessment = assessFitCore(bodyProfile, current);
    const rec = recommendSizeCore(currentAssessment, current, altProfiles, bodyProfile);

    res.json({ success: true, recommendation: rec || null, current: currentAssessment });
  } catch (err) {
    logger.error('POST /api/fit/recommend failed:', err.message);
    res.status(500).json({ error: 'Failed to recommend size' });
  }
});

// ─── helpers ────────────────────────────────────────────────────────────────

function rowToBodyProfile(row) {
  return {
    id: row.id,
    userId: row.user_id,
    status: row.status,
    isActive: row.is_active,
    gender: row.gender,
    bodyType: row.body_type,
    height: {
      valueCm: row.height_value_cm,
      confidence: row.height_confidence || 'medium',
      source: row.height_source || 'manual',
    },
    weightKg: row.weight_kg,
    measurements: row.measurements || {},
    version: row.version,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

import { recommendSize as recommendSizeCore } from '../services/fitEngine.js';

export default router;
