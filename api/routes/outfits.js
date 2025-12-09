import express from "express";
import User from "../models/user.js";
import SavedOutfit from "../models/savedoutfit.js";
import { authenticateToken } from "../middleware/auth.js";

const router = express.Router();

/**
 * POST /save-outfit
 * Save a new outfit for the authenticated user
 */
router.post("/", authenticateToken, async (req, res) => {
    try {
        const { date, items, caption, occasion, visibility, isOotd } = req.body;
        const userId = req.user.id;

        let user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ error: "User not found" });
        }

        const itemsWithImages = items?.map((item) => {
            if (!item || typeof item !== "object") {
                console.warn("Invalid item skipped", item);
                return null;
            }
            let imageUrl = item?.image;
            if (!imageUrl || !imageUrl.match(/^https?:\/\/res\.cloudinary\.com/)) {
                console.warn("Invalid or non-Cloudinary image URL:", imageUrl);
                return null;
            }
            return {
                id: item.id !== undefined || "null",
                type: item.type || "Unknown",
                image: imageUrl,
                x: item.x !== undefined ? item?.x : 0,
                y: item.y !== undefined ? item?.y : 0,
            };
        });

        const validItems = itemsWithImages.filter((item) => item !== null);

        if (validItems.length === 0) {
            return res.status(400).json({ error: "No valid items provided" });
        }

        const newOutfit = new SavedOutfit({
            userId: user._id,
            date,
            items: validItems,
            caption: caption || "",
            occasion: occasion || "",
            visibility: visibility || "Everyone",
            isOotd: isOotd || false,
        });

        await newOutfit.save();

        user.outfits.push(newOutfit._id);
        await user.save();

        console.log("✅ Outfit saved for user:", userId);
        res.status(201).json({ outfit: newOutfit });
    } catch (err) {
        console.error("Error in save-outfit:", err.message);
        res.status(500).json({ error: "Internal server error", details: err.message });
    }
});

/**
 * GET /save-outfit/user/:userId
 * Get all outfits for a specific user
 */
router.get("/user/:userId", authenticateToken, async (req, res) => {
    try {
        const userId = req.params.userId;

        if (req.user.id !== userId) {
            return res.status(403).json({ error: "Unauthorized access" });
        }

        const user = await User.findById(userId).populate("outfits");
        if (!user) {
            return res.status(404).json({ error: "User not found" });
        }

        res.status(200).json(user.outfits);
    } catch (error) {
        console.error("Error fetching outfits:", error);
        res.status(500).json({ error: "Internal server error", details: error.message });
    }
});

export default router;
