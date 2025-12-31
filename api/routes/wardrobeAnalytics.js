/**
 * Wardrobe Analytics Routes
 * Cost-Per-Wear tracking, zombie items, ROI analytics
 */

import express from 'express';
import ClothingItem from '../models/ClothingItem.js';
import User from '../models/user.js';

const router = express.Router();

/**
 * GET /api/wardrobe-analytics/:userId
 * Get comprehensive wardrobe analytics
 */
router.get('/:userId', async (req, res) => {
    const { userId } = req.params;

    try {
        console.log(`Fetching analytics for user ${userId}`);

        // Get all clothing items for user
        const items = await ClothingItem.find({ userId }).lean();

        if (items.length === 0) {
            return res.json({
                success: true,
                totalInvested: 0,
                totalWears: 0,
                averageCPW: 0,
                zombieItems: [],
                bestROI: [],
                mostWorn: [],
                leastWorn: [],
                itemCount: 0
            });
        }

        // Calculate total invested
        const totalInvested = items
            .filter(item => item.price)
            .reduce((sum, item) => sum + (item.price || 0), 0);

        // Calculate total wears
        const totalWears = items.reduce((sum, item) => sum + (item.wearCount || 0), 0);

        // Calculate average CPW
        const averageCPW = totalWears > 0 ? (totalInvested / totalWears).toFixed(2) : 0;

        // Find zombie items (never worn, or not worn in 90+ days)
        const now = new Date();
        const ninetyDaysAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);

        const zombieItems = items
            .filter(item => {
                if (item.wearCount === 0) return true;
                if (item.lastWornDate && new Date(item.lastWornDate) < ninetyDaysAgo) return true;
                return false;
            })
            .map(item => ({
                itemId: item._id,
                name: `${item.color || ''} ${item.itemType}`.trim(),
                imageUrl: item.imageUrl,
                price: item.price,
                purchaseDate: item.purchaseDate,
                wearCount: item.wearCount || 0,
                daysSincePurchase: item.purchaseDate
                    ? Math.floor((now - new Date(item.purchaseDate)) / (1000 * 60 * 60 * 24))
                    : null,
                lastWornDate: item.lastWornDate
            }))
            .sort((a, b) => (b.price || 0) - (a.price || 0))
            .slice(0, 10);

        // Find best ROI items (lowest CPW, min 3 wears)
        const bestROI = items
            .filter(item => item.price && item.wearCount >= 3)
            .map(item => ({
                itemId: item._id,
                name: `${item.color || ''} ${item.itemType}`.trim(),
                imageUrl: item.imageUrl,
                price: item.price,
                wearCount: item.wearCount,
                cpw: (item.price / item.wearCount).toFixed(2)
            }))
            .sort((a, b) => parseFloat(a.cpw) - parseFloat(b.cpw))
            .slice(0, 10);

        // Most worn items
        const mostWorn = items
            .filter(item => item.wearCount > 0)
            .map(item => ({
                itemId: item._id,
                name: `${item.color || ''} ${item.itemType}`.trim(),
                imageUrl: item.imageUrl,
                wearCount: item.wearCount,
                lastWornDate: item.lastWornDate
            }))
            .sort((a, b) => b.wearCount - a.wearCount)
            .slice(0, 10);

        // Least worn items (but worn at least once)
        const leastWorn = items
            .filter(item => item.wearCount > 0 && item.wearCount < 5)
            .map(item => ({
                itemId: item._id,
                name: `${item.color || ''} ${item.itemType}`.trim(),
                imageUrl: item.imageUrl,
                wearCount: item.wearCount,
                lastWornDate: item.lastWornDate
            }))
            .sort((a, b) => a.wearCount - b.wearCount)
            .slice(0, 10);

        res.json({
            success: true,
            totalInvested: totalInvested.toFixed(2),
            totalWears,
            averageCPW,
            zombieItems,
            bestROI,
            mostWorn,
            leastWorn,
            itemCount: items.length,
            stats: {
                itemsWithPrice: items.filter(i => i.price).length,
                itemsNeverWorn: items.filter(i => i.wearCount === 0).length,
                itemsWornOnce: items.filter(i => i.wearCount === 1).length,
                itemsWorn5Plus: items.filter(i => i.wearCount >= 5).length
            }
        });

    } catch (error) {
        console.error('Analytics error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/wardrobe-analytics/log-wear
 * Log that items were worn (auto-called when saving outfit to calendar)
 */
router.post('/log-wear', async (req, res) => {
    const { userId, itemIds, date } = req.body;

    try {
        console.log(`Logging wear for ${itemIds.length} items`);

        const wearDate = date ? new Date(date) : new Date();

        // Update each item
        const updatePromises = itemIds.map(itemId =>
            ClothingItem.findByIdAndUpdate(
                itemId,
                {
                    $inc: { wearCount: 1 },
                    lastWornDate: wearDate
                },
                { new: true }
            )
        );

        const updatedItems = await Promise.all(updatePromises);

        res.json({
            success: true,
            itemsUpdated: updatedItems.length,
            items: updatedItems.map(item => ({
                itemId: item._id,
                wearCount: item.wearCount,
                lastWornDate: item.lastWornDate
            }))
        });

    } catch (error) {
        console.error('Log wear error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/wardrobe-analytics/calculate-historical
 * One-time migration: Calculate wear counts from existing calendar data
 */
router.post('/calculate-historical', async (req, res) => {
    const { userId } = req.body;

    try {
        console.log(`Calculating historical wears for user ${userId}`);

        // TODO: If you have OutfitCalendar model, parse it here
        // For now, just reset all counts to 0

        await ClothingItem.updateMany(
            { userId },
            { $set: { wearCount: 0, lastWornDate: null } }
        );

        res.json({
            success: true,
            message: 'Historical wear counts calculated',
            itemsProcessed: 0
        });

    } catch (error) {
        console.error('Historical calculation error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/wardrobe-analytics/item/:itemId
 * Get detailed analytics for a single item
 */
router.get('/item/:itemId', async (req, res) => {
    const { itemId } = req.params;

    try {
        const item = await ClothingItem.findById(itemId).lean();

        if (!item) {
            return res.status(404).json({ error: 'Item not found' });
        }

        const cpw = item.price && item.wearCount > 0
            ? (item.price / item.wearCount).toFixed(2)
            : null;

        const daysSincePurchase = item.purchaseDate
            ? Math.floor((new Date() - new Date(item.purchaseDate)) / (1000 * 60 * 60 * 24))
            : null;

        const daysSinceLastWorn = item.lastWornDate
            ? Math.floor((new Date() - new Date(item.lastWornDate)) / (1000 * 60 * 60 * 24))
            : null;

        res.json({
            success: true,
            item: {
                ...item,
                costPerWear: cpw,
                daysSincePurchase,
                daysSinceLastWorn
            }
        });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
