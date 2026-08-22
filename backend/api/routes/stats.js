import express from "express";
import { supabase } from "../lib/supabase.js";
import { authenticateToken } from "../middleware/auth.js";

import logger from '../utils/logger.js';
const router = express.Router();

/**
 * GET /stats
 * Get wardrobe statistics for the authenticated user
 */
router.get("/", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        // Emulate ClothingItem.getStats() using Supabase
        const { data: items, error: itemsError } = await supabase
            .from('clothing_items')
            .select('category, wear_count, price')
            .eq('user_id', userId);

        if (itemsError) throw itemsError;

        const { count: outfitsCount, error: outfitsError } = await supabase
            .from('outfits')
            .select('*', { count: 'exact', head: true })
            .eq('user_id', userId);

        if (outfitsError) throw outfitsError;

        const totalItems = items.length;
        const itemsWithPrice = items.filter(i => i.price > 0);
        const totalValue = itemsWithPrice.reduce((sum, item) => sum + item.price, 0);

        // Distribution
        const distribution = items.reduce((acc, item) => {
            acc[item.category] = (acc[item.category] || 0) + 1;
            return acc;
        }, {});

        const stats = {
            totalItems,
            totalOutfits: outfitsCount || 0,
            totalValue,
            categoryDistribution: distribution
        };

        res.json({
            success: true,
            data: stats
        });
    } catch (error) {
        logger.error("Stats error:", error.message || error);
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
        const userId = req.user.id;

        const { data: items, error } = await supabase
            .from('clothing_items')
            .select('id, type, color, image_url, wear_count, last_worn, price')
            .eq('user_id', userId)
            .gt('wear_count', 0)
            .order('wear_count', { ascending: false })
            .limit(limit);

        if (error) throw error;

        // Map column names back to camelCase for the frontend
        const mappedItems = (items || []).map(i => ({
            _id: i.id,
            type: i.type,
            color: i.color,
            imageUrl: i.image_url,
            wearCount: i.wear_count,
            lastWorn: i.last_worn,
            price: i.price,
            costPerWear: i.price && i.wear_count ? (i.price / i.wear_count).toFixed(2) : 0
        }));

        res.json({ success: true, data: mappedItems });
    } catch (error) {
        logger.error("Most worn error:", error.message || error);
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
        const userId = req.user.id;

        const { data: items, error } = await supabase
            .from('clothing_items')
            .select('id, type, color, image_url, wear_count, last_worn, created_at, price')
            .eq('user_id', userId)
            // assuming we don't have an isArchived column yet in Supabase, normally we would filter here
            .order('wear_count', { ascending: true })
            .order('created_at', { ascending: true })
            .limit(limit);

        if (error) throw error;

        const mappedItems = (items || []).map(i => ({
            _id: i.id,
            type: i.type,
            color: i.color,
            imageUrl: i.image_url,
            wearCount: i.wear_count,
            lastWorn: i.last_worn,
            createdAt: i.created_at,
            price: i.price
        }));

        res.json({ success: true, data: mappedItems });
    } catch (error) {
        logger.error("Least worn error:", error.message || error);
        res.status(500).json({ error: "Failed to fetch least worn items" });
    }
});

/**
 * GET /stats/never-worn
 * Get items that have never been worn
 */
router.get("/never-worn", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        const { data: items, error } = await supabase
            .from('clothing_items')
            .select('id, type, color, image_url, created_at, price')
            .eq('user_id', userId)
            .eq('wear_count', 0)
            .order('created_at', { ascending: true });

        if (error) throw error;

        const mappedItems = (items || []).map(i => ({
            _id: i.id,
            type: i.type,
            color: i.color,
            imageUrl: i.image_url,
            createdAt: i.created_at,
            price: i.price,
            purchaseDate: i.created_at // placeholder
        }));

        res.json({
            success: true,
            data: mappedItems,
            count: mappedItems.length
        });
    } catch (error) {
        logger.error("Never worn error:", error.message || error);
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
        const sortOrder = req.query.sort === 'worst' ? -1 : 1;
        const userId = req.user.id;

        // Since Postgres doesn't easily sort by a computed cost-per-wear natively without a view/column,
        // we'll fetch elements with a price and compute/sort locally in Node.
        const { data: rawItems, error } = await supabase
            .from('clothing_items')
            .select('id, type, color, image_url, wear_count, price, brand')
            .eq('user_id', userId)
            .gt('price', 0)
            .gt('wear_count', 0);

        if (error) throw error;

        let items = (rawItems || []).map(i => ({
            _id: i.id,
            type: i.type,
            color: i.color,
            imageUrl: i.image_url,
            wearCount: i.wear_count,
            price: i.price,
            brand: i.brand,
            costPerWear: i.price / i.wear_count
        }));

        items.sort((a, b) => sortOrder === -1 ? b.costPerWear - a.costPerWear : a.costPerWear - b.costPerWear);
        items = items.slice(0, limit);

        res.json({ success: true, data: items });
    } catch (error) {
        logger.error("Cost per wear error:", error.message || error);
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
        const userId = req.user.id;

        if (!clothingItemId) {
            return res.status(400).json({ error: "clothingItemId is required" });
        }

        // RPC to increment wear count atomically
        const { data: wearLog, error: insertError } = await supabase
            .from('wear_logs')
            .insert([{
                user_id: userId,
                item_id: clothingItemId,
                outfit_id: outfitId || null,
                date: date ? new Date(date).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
                feedback: notes || null,
            }])
            .select()
            .single();

        if (insertError) throw insertError;

        // Note: In Postgres we would ideally have a DB trigger to increment item wear_count,
        // but since we are porting quickly, we'll do an explicit update.
        // First get current
        const { data: currentItem, error: getErr } = await supabase
            .from('clothing_items')
            .select('wear_count, price')
            .eq('id', clothingItemId)
            .single();

        if (!getErr && currentItem) {
            await supabase
                .from('clothing_items')
                .update({
                    wear_count: currentItem.wear_count + 1,
                    last_worn: new Date().toISOString()
                })
                .eq('id', clothingItemId);
        }

        res.json({
            success: true,
            data: {
                item: {
                    id: clothingItemId,
                    wearCount: (currentItem?.wear_count || 0) + 1,
                    costPerWear: currentItem?.price ? (currentItem.price / ((currentItem.wear_count || 0) + 1)) : 0
                },
                log: wearLog
            }
        });
    } catch (error) {
        logger.error("Log wear error:", error.message || error);
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
        const userId = req.user.id;

        if (!year || !month || month < 1 || month > 12) {
            return res.status(400).json({ error: "Invalid year or month" });
        }

        // We format standard YYYY-MM
        const monthStr = month.toString().padStart(2, '0');
        const startOfMonth = `${year}-${monthStr}-01`;
        const endOfMonth = new Date(year, month, 0).toISOString().split('T')[0]; // last day

        const { data: logs, error } = await supabase
            .from('wear_logs')
            .select(`
                id, date, feedback,
                clothing_items (id, image_url, type)
            `)
            .eq('user_id', userId)
            .gte('date', startOfMonth)
            .lte('date', endOfMonth);

        if (error) throw error;

        // Map into simple array/object grouped by date string
        const calendarData = {};
        for (const log of (logs || [])) {
            const dateKey = log.date;
            if (!calendarData[dateKey]) calendarData[dateKey] = [];
            calendarData[dateKey].push({
                _id: log.id,
                item: log.clothing_items,
                notes: log.feedback
            });
        }

        res.json({ success: true, data: calendarData, month, year });
    } catch (error) {
        logger.error("Calendar error:", error.message || error);
        res.status(500).json({ error: "Failed to fetch calendar data" });
    }
});

/**
 * GET /stats/history
 * Get wear history
 */
router.get("/history", authenticateToken, async (req, res) => {
    try {
        const { limit } = req.query;
        const userId = req.user.id;

        const { data: logs, error } = await supabase
            .from('wear_logs')
            .select(`
                id, date, feedback,
                clothing_items (id, image_url, type, color)
            `)
            .eq('user_id', userId)
            .order('date', { ascending: false })
            .limit(parseInt(limit) || 30);

        if (error) throw error;

        res.json({ success: true, data: logs || [] });
    } catch (error) {
        logger.error("History error:", error.message || error);
        res.status(500).json({ error: "Failed to fetch wear history" });
    }
});

/**
 * GET /stats/wardrobe-value
 * Get total wardrobe value and breakdown
 */
router.get("/wardrobe-value", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const { data: items, error } = await supabase
            .from('clothing_items')
            .select('category, price')
            .eq('user_id', userId)
            .gt('price', 0);

        if (error) throw error;

        let totalValue = 0;
        let totalItems = items.length;
        const byCategoryObj = {};

        for (const item of items) {
            totalValue += item.price;
            if (!byCategoryObj[item.category]) {
                byCategoryObj[item.category] = { _id: item.category, totalValue: 0, count: 0 };
            }
            byCategoryObj[item.category].totalValue += item.price;
            byCategoryObj[item.category].count += 1;
        }

        const byCategory = Object.values(byCategoryObj).map(c => ({
            ...c,
            avgPrice: c.count > 0 ? c.totalValue / c.count : 0
        })).sort((a, b) => b.totalValue - a.totalValue);

        res.json({
            success: true,
            data: {
                totalValue,
                totalItems,
                avgItemValue: totalItems > 0 ? parseFloat((totalValue / totalItems).toFixed(2)) : 0,
                byCategory
            }
        });
    } catch (error) {
        logger.error("Wardrobe value error:", error.message || error);
        res.status(500).json({ error: "Failed to calculate wardrobe value" });
    }
});

export default router;
