import mongoose from "mongoose";

const ClothingItemSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  type: { type: String, required: true }, // Например: "Shirt"
  color: { type: String },                // "Blue"
  season: { type: String },               // "Summer"
  style: { type: String },                // "Casual"
  description: { type: String },          // "Light blue linen shirt"
  
  // Важный момент: Gemini 1.5 Flash по видео дает только текст.
  // Пока поставим заглушку или будем использовать скриншот позже.
  imageUrl: { type: String, default: "https://via.placeholder.com/150" }, 
  
  createdAt: { type: Date, default: Date.now },
});

const ClothingItem = mongoose.model("ClothingItem", ClothingItemSchema);
export default ClothingItem;