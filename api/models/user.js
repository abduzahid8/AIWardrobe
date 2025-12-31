import mongoose from "mongoose";
import bcrypt from "bcryptjs";

const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true, lowercase: true },
  password: { type: String, required: true },
  username: { type: String, required: true, unique: true },
  gender: String,
  profileImage: String,
  outfits: [{ type: mongoose.Schema.Types.ObjectId, ref: "SavedOutfit" }],

  // Gmail OAuth tokens for email receipt ingestion (encrypted in production)
  gmailRefreshToken: String,
  gmailAccessToken: String,
  gmailTokenExpiry: Date,

  // Security: Account lockout
  failedLoginAttempts: { type: Number, default: 0 },
  lockedUntil: Date,
  lastFailedLogin: Date,

  // Security: Login tracking
  lastLoginAt: Date,
  lastLoginIP: String,

  // Subscription (cached for quick access)
  subscriptionTier: {
    type: String,
    enum: ['free', 'premium', 'vip'],
    default: 'free'
  },
  subscriptionExpiresAt: Date,

  // Account status
  isActive: { type: Boolean, default: true },
  isEmailVerified: { type: Boolean, default: false },
  emailVerificationToken: String,
  passwordResetToken: String,
  passwordResetExpires: Date,

  // Timestamps
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
}, {
  timestamps: true
});

// Indexes for performance
userSchema.index({ email: 1 });
userSchema.index({ subscriptionTier: 1 });

userSchema.pre("save", async function (next) {
  if (this.isModified("password")) {
    this.password = await bcrypt.hash(this.password, 12); // Increased from 10 to 12 rounds
  }
  next();
});

userSchema.methods.comparePassword = async function (password) {
  return bcrypt.compare(password, this.password);
};

/**
 * Check if account is currently locked
 */
userSchema.methods.isLocked = function () {
  return this.lockedUntil && this.lockedUntil > new Date();
};

/**
 * Check if user has premium access
 */
userSchema.methods.hasPremiumAccess = function () {
  if (!['premium', 'vip'].includes(this.subscriptionTier)) {
    return false;
  }
  if (this.subscriptionExpiresAt && this.subscriptionExpiresAt < new Date()) {
    return false;
  }
  return true;
};

export default mongoose.model("User", userSchema);

