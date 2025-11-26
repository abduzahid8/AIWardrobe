import mongoose from "mongoose";

const outfitSchema = new mongoose.Schema({
    ocassion:String,
    style:String,
    items:[String],
    image:String,
    embedding:[Number]
});

export default mongoose.model("Outfit",outfitSchema);