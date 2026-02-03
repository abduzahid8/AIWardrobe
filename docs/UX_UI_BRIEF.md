# AIWardrobe - UX/UI Design Brief

## 1. Project Overview
**Product Name:** AIWardrobe
**Platform:** Mobile App (iOS & Android) - Built with React Native & Expo
**Core Value:** A smart AI-powered wardrobe management app that allows users to scan their clothes via video, get personalized outfit suggestions, and virtually try on clothes. It solves "wardrobe paralysis" by digitizing the user's closet and acting as a personal AI stylist.

## 2. Target Audience
- Fashion-conscious individuals who own many clothes but struggle to put outfits together.
- Users who want to organize their wardrobe digitally.
- People looking for style inspiration and "smart" fashion advice.

## 3. Design Aesthetic & Visual Language
The app aims for a **premium, futuristic, and highly interactive** feel. The current codebase suggests a sophisticated UI with the following elements:
- **Glassmorphism:** Extensive use of blur effects, transparency, and "liquid glass" styles (e.g., `LiquidGlassCard`).
- **Bento Grids:** Modern, grid-based layouts for dashboards and summary screens (e.g., `BentoGrid`).
- **Rich Animations:** Smooth transitions, layout animations (Reanimated), and skeletons (`LoadingState`).
- **Dark Mode/High Contrast:** Likely focus on sleek, dark-themed interfaces to make clothing images pop (inferred from modern design trends in this space).

## 4. Key Features
### A. Intelligent Wardrobe Scanning
- **Video Scanning:** Users scan their closet using video.
- **AI Processing:** The app uses SegFormer and MediaPipe to detect, segment, and categorize items from the video feed.
- **Product Cards:** Generates professional e-commerce style images with background removal.

### B. Virtual Try-On
- Users can virtually try on items using AI generation (Replicate/Gemini).
- "Magic Mirror" functionality.

### C. AI Stylist & Chat
- Conversational interface (`AIAssistant`, `ChatBubble`) for style advice.
- Outfit generation based on weather (`Weather-Based Suggestions`) and occasions (`MeetingOutfit`, `TripPlanner`).

### D. Digital Closet Management
- Categorized view of all items (`MyClosetScreen`).
- Manual and AI tagging of attributes (Color, Pattern, Material).

## 5. Core Screens & User Flows
Based on the current project structure, the following key screens need design attention:

### Onboarding & Auth
- **Welcome/Auth:** `SignInScreen`, `SignUpScreen`, `EmailOnboardingScreen`
- **Profiling:** `StyleQuizScreen`, `StyleGoalsScreen`, `BrandSelectionScreen`

### Main Functionality
- **Home:** `HomeScreen` - Dashboard with quick actions and daily suggestions.
- **Wardrobe:** `MyClosetScreen` - Grid view of items, `OutfitDetailScreen`.
- **Scanning:** `CameraScreen`, `WardrobeVideoScreen`, `ReviewScreen` - The core "input" flow.
- **Creation:** `DesignRoomScreen`, `NewOutfitScreen` - Canvas for mixing and matching.
- **AI Features:** `OutfitAIScreen`, `MagicMirrorScreen`, `AITryOnScreen`.

### Utilities & Planning
- `TripPlannerScreen` - Packing lists.
- `OutfitCalendarScreen` - Scheduling outfits.
- `PriceTrackerScreen` / `FlashSalesScreen` - Shopping integration.
- `ProfileScreen` - User settings and stats.

## 6. Existing Design Components
The developer implementation includes these specific reusable components which define the current "look & feel":
- **`LiquidGlassCard`**: A signature component probably used for highlighting premium features or items.
- **`BentoGrid`**: Used for dashboard layouts.
- **`ActionCard`**, **`ClothingCard`**, **`ProductCard`**: Standard list items.
- **`CelebrityClothingCard`**: Social proof/inspiration element.
- **`SwipeableClothingCarousel`**: Interactive browsing.

## 7. Technical Constraints & Considerations
- **React Native / Expo:** Designs must be implementable in RN. Complex shadows and blurs can be performance-heavy on Android; require optimization.
- **Safe Areas:** Design must account for notches and home indicators (`ScreenWrapper` handles this).
- **Navigation:** Uses Native Stack and Bottom Tabs (`navigation/` directory).
- **Assets:** SVG icons (`@expo/vector-icons`) and remote images (Supabase/Cloudinary).

## 8. Deliverables Required regarding Design
1. **High-Fidelity Mockups:** For all core screens.
2. **Design System:** Colors, Typography, Spacing, and Component Library (Buttons, Inputs, Cards).
3. **Prototypes:** Demonstrating the "Liquid" and scanning interactions.
4. **Assets:** Exported icons and illustrations.
