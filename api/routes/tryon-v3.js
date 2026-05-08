/**
 * POST /api/tryon/render-v3
 *
 * Pose-Anchored Multi-Reference Conditioning (v3) try-on route.
 *
 * This is an explicit endpoint for the v3 strategy, independent of the
 * TRYON_STRATEGY env flag. The main /api/tryon/render route also
 * dispatches here when TRYON_STRATEGY=v3.
 *
 * Body: same as /api/tryon/render
 * Response: same as /api/tryon/render
 */

import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import logger from '../utils/logger.js';
import { poseAnchoredRender } from '../services/strategies/poseAnchored.js';

const router = express.Router();

router.post('/render-v3', authenticateToken, async (req, res) => {
  try {
    const result = await poseAnchoredRender(req.body);

    if (!result.success) {
      return res.status(400).json(result);
    }

    return res.json(result);
  } catch (err) {
    logger.error('[tryon/render-v3] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'v3 render failed',
    });
  }
});

export default router;
