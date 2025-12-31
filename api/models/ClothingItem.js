import mongoose from "mongoose";

const ClothingItemSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: false,  // Made optional for video scan items
    index: true,
  },

  sourceMetadata: mongoose.Schema.Types.Mixed, // Added sourceMetadata field

  // Basic Info
  type: { type: String, required: true },  // e.g., "Shirt", "Jeans"
  category: {
    type: String,
    enum: ['Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Accessories', 'Other'],
    default: 'Other'
  },
  color: { type: [String], default: [] },  // Support multiple colors
  style: {
    type: String,
    enum: ['Casual', 'Formal', 'Sport', 'Streetwear', 'Beach', 'Elegant', 'Business', 'Other'],
    default: 'Casual'
  },

  // Additional Details
  brand: { type: String, default: '' },
  material: { type: String, default: '' },  // Cotton, Denim, Leather, etc.
  pattern: {
    type: String,
    enum: ['Solid', 'Striped', 'Checkered', 'Floral', 'Printed', 'Other'],
    default: 'Solid'
  },

  // Seasons & Occasions
  season: {
    type: [String],
    enum: ['Spring', 'Summer', 'Fall', 'Winter', 'All Seasons'],
    default: ['All Seasons']
  },
  occasion: {
    type: [String],
    default: []  // e.g., ['Work', 'Casual', 'Party', 'Date']
  },

  // Cost & Value Tracking
  price: { type: Number, default: 0 },
  currency: { type: String, default: 'USD' },
  purchaseDate: { type: Date },
  purchaseLocation: { type: String, default: '' },

  // Wear Tracking for analytics
  wearCount: { type: Number, default: 0 },
  lastWornDate: { type: Date }, // Renamed from lastWorn to lastWornDate

  // Computed field (updated on each wear)
  costPerWear: { type: Number, default: 0 },

  // Media
  imageUrl: { type: String, default: "https://via.placeholder.com/150" },
  thumbnailUrl: { type: String },

  // Organization
  isFavorite: { type: Boolean, default: false },
  isArchived: { type: Boolean, default: false },
  notes: { type: String, default: '' },
  tags: { type: [String], default: [] },

  // AI-generated description
  description: { type: String, default: '' },
  aiGenerated: { type: Boolean, default: false },

}, {
  timestamps: true, // Adds createdAt and updatedAt
});

// Index for efficient queries
ClothingItemSchema.index({ userId: 1, category: 1 });
ClothingItemSchema.index({ userId: 1, isFavorite: 1 });
ClothingItemSchema.index({ userId: 1, wearCount: 1 });
ClothingItemSchema.index({ userId: 1, lastWornDate: 1 }); // Updated index to lastWornDate

// Virtual field for cost-per-wear calculation
ClothingItemSchema.virtual('calculatedCostPerWear').get(function () {
  if (this.wearCount === 0 || !this.price) return null;
  return (this.price / this.wearCount).toFixed(2);
});

// Virtual field for cost-per-wear calculation (as requested by instruction)
ClothingItemSchema.virtual('costPerWearVirtual').get(function () {
  if (!this.price || this.wearCount === 0) return null;
  return (this.price / this.wearCount).toFixed(2);
});

// Method to log a wear
ClothingItemSchema.methods.logWear = async function () {
  this.wearCount += 1;
  this.lastWornDate = new Date(); // Updated to lastWornDate
  if (this.price > 0) {
    this.costPerWear = parseFloat((this.price / this.wearCount).toFixed(2));
  }
  return this.save();
};

// Static method to get wardrobe statistics
ClothingItemSchema.statics.getStats = async function (userId) {
  const stats = await this.aggregate([
    { $match: { userId: new mongoose.Types.ObjectId(userId) } },
    {
      $group: {
        _id: null,
        totalItems: { $sum: 1 },
        totalValue: { $sum: '$price' },
        avgWearCount: { $avg: '$wearCount' },
        neverWorn: {
          $sum: { $cond: [{ $eq: ['$wearCount', 0] }, 1, 0] }
        },
        favorites: {
          $sum: { $cond: ['$isFavorite', 1, 0] }
        },
      }
    }
  ]);

  const categoryStats = await this.aggregate([
    { $match: { userId: new mongoose.Types.ObjectId(userId) } },
    { $group: { _id: '$category', count: { $sum: 1 } } }
  ]);

  const colorStats = await this.aggregate([
    { $match: { userId: new mongoose.Types.ObjectId(userId) } },
    { $unwind: '$color' },
    { $group: { _id: '$color', count: { $sum: 1 } } },
    { $sort: { count: -1 } },
    { $limit: 10 }
  ]);

  return {
    ...(stats[0] || { totalItems: 0, totalValue: 0, avgWearCount: 0, neverWorn: 0, favorites: 0 }),
    byCategory: categoryStats.reduce((acc, item) => {
      acc[item._id || 'Other'] = item.count;
      return acc;
    }, {}),
    topColors: colorStats.map(c => ({ color: c._id, count: c.count }))
  };
};

const ClothingItem = mongoose.model("ClothingItem", ClothingItemSchema, "clothingitems"); // Changed collection name
export default ClothingItem;