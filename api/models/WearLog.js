import mongoose from "mongoose";

/**
 * WearLog Model
 * Tracks when clothing items are worn for statistics and outfit history
 */
const WearLogSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        index: true,
    },

    // What was worn
    clothingItemId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "ClothingItem",
        required: true,
    },

    // Optional outfit association
    outfitId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "SavedOutfit",
    },

    // When it was worn
    date: {
        type: Date,
        required: true,
        index: true,
    },

    // Context
    occasion: { type: String, default: '' },
    weather: { type: String, default: '' },
    temperature: { type: Number },

    // User notes
    notes: { type: String, default: '' },

    // Photo of the day (optional)
    photoUrl: { type: String },

}, {
    timestamps: true,
});

// Compound index for efficient date-based queries
WearLogSchema.index({ userId: 1, date: -1 });
WearLogSchema.index({ clothingItemId: 1, date: -1 });

// Static method to get wear history for a user
WearLogSchema.statics.getHistory = async function (userId, options = {}) {
    const { startDate, endDate, limit = 30 } = options;

    const query = { userId: new mongoose.Types.ObjectId(userId) };

    if (startDate || endDate) {
        query.date = {};
        if (startDate) query.date.$gte = new Date(startDate);
        if (endDate) query.date.$lte = new Date(endDate);
    }

    return this.find(query)
        .sort({ date: -1 })
        .limit(limit)
        .populate('clothingItemId', 'type color imageUrl')
        .populate('outfitId', 'caption');
};

// Static method to get calendar data (items worn each day)
WearLogSchema.statics.getCalendarData = async function (userId, year, month) {
    const startDate = new Date(year, month - 1, 1);
    const endDate = new Date(year, month, 0);

    return this.aggregate([
        {
            $match: {
                userId: new mongoose.Types.ObjectId(userId),
                date: { $gte: startDate, $lte: endDate }
            }
        },
        {
            $group: {
                _id: { $dateToString: { format: "%Y-%m-%d", date: "$date" } },
                items: { $push: "$clothingItemId" },
                count: { $sum: 1 }
            }
        },
        { $sort: { _id: 1 } }
    ]);
};

// Static method to find repeated outfits
WearLogSchema.statics.findRepeatedOutfits = async function (userId, daysBack = 14) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - daysBack);

    // Get recent outfit IDs
    const recentOutfits = await this.find({
        userId: new mongoose.Types.ObjectId(userId),
        outfitId: { $exists: true, $ne: null },
        date: { $gte: startDate }
    }).distinct('outfitId');

    return recentOutfits;
};

const WearLog = mongoose.model("WearLog", WearLogSchema);
export default WearLog;
