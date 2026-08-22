import express from "express";
import { authenticateToken, optionalAuth } from "../middleware/auth.js";

import logger from '../utils/logger.js';
const router = express.Router();

/**
 * POST /api/analytics/events
 * Receive batched analytics events from the mobile app.
 * Stores events and optionally forwards to PostHog/Mixpanel.
 *
 * When ready to integrate PostHog:
 *   npm install posthog-node
 *   const { PostHog } = require('posthog-node');
 *   const client = new PostHog(process.env.POSTHOG_API_KEY, { host: 'https://app.posthog.com' });
 */
router.post("/events", optionalAuth, async (req, res) => {
    try {
        const { events, sessionId } = req.body;

        if (!Array.isArray(events) || events.length === 0) {
            return res.status(400).json({ error: "events array is required" });
        }

        // Cap batch size to prevent abuse
        if (events.length > 200) {
            return res.status(400).json({ error: "Maximum 200 events per batch" });
        }

        const userId = req.user?.id || 'anonymous';

        // Log events in structured format for log aggregation
        for (const event of events) {
            logger.info(JSON.stringify({
                type: 'analytics',
                name: event.name,
                userId,
                sessionId: sessionId || event.sessionId,
                timestamp: event.timestamp,
                properties: event.properties || {},
            }));
        }

        // TODO: When PostHog is configured, forward events:
        // for (const event of events) {
        //     posthogClient.capture({
        //         distinctId: userId,
        //         event: event.name,
        //         properties: { ...event.properties, sessionId },
        //         timestamp: new Date(event.timestamp),
        //     });
        // }

        res.json({
            success: true,
            received: events.length,
        });
    } catch (error) {
        logger.error("Analytics ingestion error:", error.message);
        res.status(500).json({ error: "Failed to process events" });
    }
});

/**
 * GET /api/analytics/summary
 * Get basic analytics summary for the user.
 * Protected — requires auth.
 */
router.get("/summary", authenticateToken, async (req, res) => {
    try {
        // For now, return a placeholder. When PostHog/Mixpanel is configured,
        // query their API for user-specific metrics.
        res.json({
            success: true,
            message: "Analytics summary will be available when PostHog is configured.",
        });
    } catch (error) {
        logger.error("Analytics summary error:", error.message);
        res.status(500).json({ error: "Failed to get analytics" });
    }
});

export default router;
