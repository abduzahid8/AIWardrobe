# App Review Notes Template (AIWardrobe)

Copy/paste this into **App Store Connect → App Review → Notes** and adjust the bracketed parts.

## Demo account (if needed)
- **Email**: [reviewer+aiwardrobe@example.com]
- **Password**: [password]
- **Notes**: Sign In is required to use the app. You can sign in with Apple or email/password.

## What the app does
AIWardrobe lets users:
- Add clothing items to a digital wardrobe (camera or photo library)
- Run AI wardrobe scan to detect items and metadata (category/color/material)
- Generate outfit recommendations and calendar planning
- Create AI "try-on" previews on a mannequin using selected shop/wardrobe items
- Browse a curated shop catalog with affiliate links to physical goods
- Manage a subscription via Apple In‑App Purchase (RevenueCat)

## Key feature test steps
### 1) Wardrobe scan (camera / gallery)
- Go to **Closet** tab → tap **+** to add clothing
- Take a photo or pick from gallery → review AI-detected metadata → **Save**

### 2) AI Try‑On (mannequin)
- Go to **Home** tab → tap **Try On**
- Pick 1–4 pieces (Top, Layer, Pants, Shoes)
- Tap **Try On With AI**
- Tap **Save Look** to save the result

### 3) Subscription
- Open **Paywall** from Profile → **Subscription** or from any feature gate
- Two plans: **Pro** (monthly) and **Max** (yearly)
- Purchase using Apple IAP via StoreKit/RevenueCat
- Use **Restore Purchases** to verify restore flow
- Use **Manage Subscription** to open Apple subscription management
- **Redeem Offer Code**: tap the button to open Apple's native offer code redemption sheet (iOS 14+)

## Permissions (why we ask)
- **Camera**: capture clothing photos/videos you choose to add
- **Photos**: import selected images and save/share generated outfit images (only when you choose)
- **Microphone**: only used when recording a wardrobe video (not required for photos)
- **Location (When In Use)**: optional; used to fetch local weather for weather‑appropriate suggestions (not stored or shared)

## Data & AI processing disclosure
- Images selected by the user may be uploaded to our backend to run AI processing (classification, background removal, try-on rendering).
- We use third‑party service providers to deliver these features (AI + infrastructure). We do not sell personal data.

## Account deletion (in‑app)
- Go to **Profile** → scroll down → **Delete Account** → confirm **Delete Everything**
- This permanently deletes the user's account and all associated wardrobe data/media.
- Data export is also available: Profile → request data export (GDPR right to portability)

## Sign in with Apple
- Available on the Sign In screen alongside email/password authentication.
