import express from "express";
import ClothingItem from "../models/ClothingItem.js";
import WearLog from "../models/WearLog.js";
import { authenticateToken } from "../middleware/auth.js";

const router = express.Router();

/**
 * GET /stats
 * Get wardrobe statistics for the authenticated user
 */
router.get("/", authenticateToken, async (req, res) => {
    try {
        const stats = await ClothingItem.getStats(req.user.id);
        res.json({
            success: true,
            data: stats
        });
    } catch (error) {
        console.error("Stats error:", error.message);
        res.status(500).json({
            error: "Failed to fetch statistics",
            details: error.message
        });
    }
});

/**
 * GET /stats/most-worn
 * Get most worn items
 */
router.get("/most-worn", authenticateToken, async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 10;

        const items = await ClothingItem.find({
            userId: req.user.id,
            wearCount: { $gt: 0 }
        })
            .sort({ wearCount: -1 })
            .limit(limit)
            .select('type color imageUrl wearCount lastWorn costPerWear price');

        res.json({
            success: true,
            data: items
        });
    } catch (error) {
        console.error("Most worn error:", error.message);
        res.status(500).json({ error: "Failed to fetch most worn items" });
    }
});

/**
 * GET /stats/least-worn
 * Get least worn items (potential candidates for donation/selling)
 */
router.get("/least-worn", authenticateToken, async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 10;

        const items = await ClothingItem.find({
            userId: req.user.id,
            isArchived: { $ne: true }
        })
            .sort({ wearCount: 1, createdAt: 1 })
            .limit(limit)
            .select('type color imageUrl wearCount lastWorn createdAt price');

        res.json({
            success: true,
            data: items
        });
    } catch (error) {
        console.error("Least worn error:", error.message);
        res.status(500).json({ error: "Failed to fetch least worn items" });
    }
});

/**
 * GET /stats/never-worn
 * Get items that have never been worn
 */
router.get("/never-worn", authenticateToken, async (req, res) => {
    try {
        const items = await ClothingItem.find({
            userId: req.user.id,
            wearCount: 0,
            isArchived: { $ne: true }
        })
            .sort({ createdAt: 1 })
            .select('type color imageUrl createdAt price purchaseDate');

        res.json({
            success: true,
            data: items,
            count: items.length
        });
    } catch (error) {
        console.error("Never worn error:", error.message);
        res.status(500).json({ error: "Failed to fetch never worn items" });
    }
});

/**
 * GET /stats/cost-per-wear
 * Get items sorted by cost per wear (best value)
 */
router.get("/cost-per-wear", authenticateToken, async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 20;
        const sortOrder = req.query.sort === 'worst' ? -1 : 1; // 'best' or 'worst' value

        const items = await ClothingItem.find({
            userId: req.user.id,
            price: { $gt: 0 },
            wearCount: { $gt: 0 }
        })
            .sort({ costPerWear: sortOrder })
            .limit(limit)
            .select('type color imageUrl wearCount costPerWear price brand');

        res.json({
            success: true,
            data: items
        });
    } catch (error) {
        console.error("Cost per wear error:", error.message);
        res.status(500).json({ error: "Failed to fetch cost per wear data" });
    }
});

/**
 * POST /stats/log-wear
 * Log that an item was worn
 */
router.post("/log-wear", authenticateToken, async (req, res) => {
    try {
        const { clothingItemId, outfitId, date, occasion, weather, notes } = req.body;

        if (!clothingItemId) {
            return res.status(400).json({ error: "clothingItemId is required" });
        }

        // Update the clothing item's wear count
        const item = await ClothingItem.findOne({
            _id: clothingItemId,
            userId: req.user.id
        });

        if (!item) {
            return res.status(404).json({ error: "Clothing item not found" });
        }

        await item.logWear();

        // Create wear log entry
        const wearLog = new WearLog({
            userId: req.user.id,
            clothingItemId,
            outfitId,
            date: date ? new Date(date) : new Date(),
            occasion,
            weather,
            notes
        });

        await wearLog.save();

        res.json({
            success: true,
            data: {
                item: {
                    id: item._id,
                    wearCount: item.wearCount,
                    costPerWear: item.costPerWear
                },
                log: wearLog
            }
        });
    } catch (error) {
        console.error("Log wear error:", error.message);
        res.status(500).json({ error: "Failed to log wear" });
    }
});

/**
 * GET /stats/calendar
 * Get calendar data for a month
 */
router.get("/calendar/:year/:month", authenticateToken, async (req, res) => {
    try {
        const year = parseInt(req.params.year);
        const month = parseInt(req.params.month);

        if (!year || !month || month < 1 || month > 12) {
            return res.status(400).json({ error: "Invalid year or month" });
        }

        const calendarData = await WearLog.getCalendarData(req.user.id, year, month);

        res.json({
            success: true,
            data: calendarData,
            month,
            year
        });
    } catch (error) {
        console.error("Calendar error:", error.message);
        res.status(500).json({ error: "Failed to fetch calendar data" });
    }
});

/**
 * GET /stats/history
 * Get wear history
 */
router.get("/history", authenticateToken, async (req, res) => {
    try {
        const { startDate, endDate, limit } = req.query;

        const history = await WearLog.getHistory(req.user.id, {
            startDate,
            endDate,
            limit: parseInt(limit) || 30
        });

        res.json({
            success: true,
            data: history
        });
    } catch (error) {
        console.error("History error:", error.message);
        res.status(500).json({ error: "Failed to fetch wear history" });
    }
});

/**
 * GET /stats/wardrobe-value
 * Get total wardrobe value and breakdown
 */
router.get("/wardrobe-value", authenticateToken, async (req, res) => {
    try {
        const valueStats = await ClothingItem.aggregate([
            { $match: { userId: req.user.id, price: { $gt: 0 } } },
            {
                $group: {
                    _id: '$category',
                    totalValue: { $sum: '$price' },
                    count: { $sum: 1 },
                    avgPrice: { $avg: '$price' }
                }
            },
            { $sort: { totalValue: -1 } }
        ]);

        const totalValue = valueStats.reduce((sum, cat) => sum + cat.totalValue, 0);
        const totalItems = valueStats.reduce((sum, cat) => sum + cat.count, 0);

        res.json({
            success: true,
            data: {
                totalValue,
                totalItems,
                avgItemValue: totalItems > 0 ? (totalValue / totalItems).toFixed(2) : 0,
                byCategory: valueStats
            }
        });
    } catch (error) {
        console.error("Wardrobe value error:", error.message);
        res.status(500).json({ error: "Failed to calculate wardrobe value" });
    }
});

export default router;
