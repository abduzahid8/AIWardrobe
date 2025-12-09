import express from "express";
import mongoose from "mongoose";
import ClothingItem from "../models/ClothingItem.js";
import User from "../models/user.js";
import Replicate from "replicate";
import axios from "axios";
import { createClient } from "@supabase/supabase-js";
import { authenticateToken } from "../middleware/auth.js";

const router = express.Router();

// Initialize Supabase
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// Initialize Replicate
const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

/**
 * POST /clothing-items
 * Save a single clothing item from video scan
 */
router.post("/", authenticateToken, async (req, res) => {
    try {
        const { type, color, style, description, season, imageUrl } = req.body;

        // Get userId from the authenticated token and convert to ObjectId
        const userId = new mongoose.Types.ObjectId(req.user.id);

        const newItem = new ClothingItem({
            userId: userId,
            type: type || "Unknown",
            color: color || "Unknown",
            style: style || "Casual",
            description: description || "",
            season: season || "All Seasons",
            imageUrl: imageUrl || "https://via.placeholder.com/150",
            createdAt: new Date(),
        });

        await newItem.save();
        console.log("✅ Saved clothing item:", newItem.type, "for user:", userId);
        res.status(201).json({ success: true, item: newItem });
    } catch (error) {
        console.error("Error saving clothing item:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /clothing-items
 * Fetch user's clothing items
 */
router.get("/", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const items = await ClothingItem.find({ userId }).sort({ createdAt: -1 });
        console.log("📦 Found", items.length, "items for user:", userId);
        res.json({ items });
    } catch (error) {
        console.error("Error fetching clothing items:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /wardrobe/add-batch
 * Bulk add clothing items with AI-generated images
 */
router.post("/add-batch", authenticateToken, async (req, res) => {
    try {
        const { items } = req.body;
        const userId = req.user.id;

        if (!items || !Array.isArray(items) || items.length === 0) {
            return res.status(400).json({ error: "No items provided" });
        }

        console.log(`🎨 Processing ${items.length} items via Supabase...`);

        const itemsWithImages = await Promise.all(
            items.map(async (item) => {
                let finalImageUrl = "https://via.placeholder.com/300?text=No+Image";

                try {
                    // Generate prompt for image
                    const prompt = `A professional studio photography of a ${item.color} ${item.style} ${item.itemType} (${item.description}), isolated on clean white background, flat lay, fashion catalog style, high quality, realistic, no shadows`;

                    // Generate image with Replicate
                    const output = await replicate.run("black-forest-labs/flux-schnell", {
                        input: {
                            prompt: prompt,
                            aspect_ratio: "1:1",
                            output_format: "jpg",
                            output_quality: 80,
                        },
                    });

                    // If image generated, upload to Supabase
                    if (output && output[0]) {
                        const replicateUrl = output[0];

                        // Download image
                        const imageResponse = await axios.get(replicateUrl, {
                            responseType: "arraybuffer",
                        });
                        const buffer = Buffer.from(imageResponse.data, "binary");

                        // Generate unique filename
                        const fileName = `${userId}/${Date.now()}_${Math.random().toString(36).substring(7)}.jpg`;

                        // Upload to Supabase Storage
                        const { data, error } = await supabase.storage
                            .from("AIWARDROBE")
                            .upload(fileName, buffer, {
                                contentType: "image/jpeg",
                                upsert: false,
                            });

                        if (error) {
                            console.error("Supabase error:", error);
                            throw error;
                        }

                        // Get public URL
                        const { data: publicUrlData } = supabase.storage
                            .from("AIWARDROBE")
                            .getPublicUrl(fileName);

                        finalImageUrl = publicUrlData.publicUrl;
                    }
                } catch (genError) {
                    console.error(`Error with item ${item.itemType}:`, genError.message);
                }

                // Return object for MongoDB
                return {
                    userId: userId,
                    type: item.itemType,
                    color: item.color,
                    season: item.season,
                    style: item.style,
                    description: item.description,
                    imageUrl: finalImageUrl,
                };
            })
        );

        // Save to MongoDB
        const savedItems = await ClothingItem.insertMany(itemsWithImages);

        // Update user
        await User.findByIdAndUpdate(userId, {
            $push: { outfits: { $each: savedItems.map((i) => i._id) } },
        });

        console.log(`✅ Successfully saved: ${savedItems.length} items`);
        res.status(201).json({ success: true, count: savedItems.length });
    } catch (err) {
        console.error("Critical Error:", err);
        res.status(500).json({ error: err.message });
    }
});

export default router;
