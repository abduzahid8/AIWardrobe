import mongoose from "mongoose";

const savedOutfitSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  date: { type: String, required: true },
  items: [
    {
      id: Number,
      type: { type: String, enum: ["pants", "shoes", "shirt"], required: true },
      image: { type: String, required: true },
      x: { type: Number, default: 0 },
      y: { type: Number, default: 0 },
    },
  ],
  caption: { type: String, default: "" },
  ocassion: { type: String, default: "casual" },
  visibilty: { type: String, default: "Everyone" },
  isOotd: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.model("SavedOutfit",savedOutfitSchema);