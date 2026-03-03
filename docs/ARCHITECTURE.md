# AIWardrobe — System Architecture

> **Core Thesis:** AIWardrobe eliminates daily outfit decision fatigue by converting a one-time video scan into a persistent wardrobe graph that generates weather-aware, style-matched, context-specific outfit suggestions — creating a daily open loop that increases closet utilization over time.

---

## 1. Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UX LAYER                                │
│  Screens, Components, Animations, Navigation                 │
│  React Native + Expo + NativeWind + Reanimated               │
│                                                              │
│  Core Screens (MVP):                                         │
│    Onboarding (StyleQuiz) → Home → DailySuggestion           │
│    → WearLog → MyCloset → WeeklyInsights → Profile           │
│                                                              │
│  v2 Screens:                                                 │
│    AIChat, VirtualTryOn, TripPlanner, OutfitCalendar         │
│                                                              │
│  Long-term:                                                  │
│    FlashSales, PriceTracker, DesignRoom, Avatar, Inspo       │
├─────────────────────────────────────────────────────────────┤
│                    LOGIC LAYER                                │
│  State Management + Business Rules + Scoring                 │
│                                                              │
│  Stores (Zustand + AsyncStorage persistence):                │
│    wardrobeStore  — items, outfits, wearLogs, suggestions    │
│    stylePreferenceStore — quiz results, feedback, learning   │
│    authStore — auth state, trial mode                        │
│    subscriptionStore — tier, paywall state                   │
│                                                              │
│  Services:                                                   │
│    suggestionEngine.ts — RULE-BASED outfit scoring           │
│      (preference × weather × novelty × color_harmony)        │
│    retentionService.ts — streaks, utilization, insights      │
│    notificationService.ts — push scheduling + navigation     │
├─────────────────────────────────────────────────────────────┤
│                      AI LAYER                                │
│  Model Inference + LLM Text Generation                       │
│                                                              │
│  visionService.ts — AliceVision client (segmentation,        │
│    detection, attribute extraction, multi-frame analysis)     │
│  llmService.ts — GPT/Gemini text generation (chat,           │
│    outfit explanations — cosmetic, not core logic)            │
│  aiService.ts — LEGACY monolith (being decomposed)           │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                 │
│  Persistence + Sync + Offline Queue                          │
│                                                              │
│  Local: Zustand + AsyncStorage (offline-first)               │
│  Remote: Supabase (PostgreSQL + Storage + Auth)              │
│  Sync: pendingActions queue → syncToServer()                 │
│                                                              │
│  Tables:                                                     │
│    users — profile, preferences, subscription                │
│    clothing_items — AI-detected + user-edited                │
│    outfits — generated + saved, with ratings                 │
│    wear_logs — behavioral data (items, occasion, weather)    │
│    daily_suggestions — cached per user per day               │
├─────────────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE LAYER                          │
│  Servers + AI Services + Storage                             │
│                                                              │
│  Express.js API (Node.js, port 3000)                         │
│    Routes: auth, clothing, outfits, weather, ai              │
│  AliceVision AI Service (Python, port 5050)                  │
│    Models: SegFormer-B2, MediaPipe, Fashion-CLIP, K-means    │
│  Supabase (PostgreSQL + S3 storage)                          │
│  External APIs: Gemini, OpenAI, Replicate, WeatherAPI        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. What Is Rule-Based vs AI-Based

| Component | Type | Rationale |
|-----------|------|-----------|
| Image segmentation | **ML Model** (SegFormer) | Requires visual understanding |
| Attribute extraction | **ML Model** (Fashion-CLIP + K-means) | Pattern/material recognition |
| Weather filtering | **Rule-based** | `if temp < 5: exclude shorts` — deterministic |
| Season filtering | **Rule-based** | Month → season mapping — deterministic |
| Occasion matching | **Rule-based** | User-tagged items × occasion filter |
| Outfit scoring | **Hybrid** | Rule-based weights + color harmony table |
| Novelty scoring | **Rule-based** | `1 / (recentWears + 1)` — pure math |
| Preference scoring | **Rule-based** | Color/pattern match from quiz results |
| Streak calculation | **Rule-based** | Consecutive date counting |
| Closet utilization | **Rule-based** | Set intersection math |
| Styling explanation | **LLM** (GPT-4/Gemini) | Natural language — cosmetic |
| AI chat stylist | **LLM** (GPT-4/Gemini) | Conversational UI |
| Virtual try-on | **ML Model** (Replicate diffusion) | Image generation |
| Outfit diversity | **Rule-based** | Overlap percentage deduplication |

**Key principle:** Core scoring is rule-based (fast, free, debuggable). LLMs are cosmetic text generators only. The system works without any LLM calls.

---

## 3. Core User Loop

```
┌─── MORNING (7:00 AM) ───────────────────────────────────────┐
│  Push: "Your outfit for today is ready"                      │
│  → Opens DailySuggestionScreen                               │
│  → 3 AI-scored options (weather + preference + novelty)      │
│  → [Wear This] / [Try Another] / [Skip]                     │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌─── EVENING (8:00 PM) ────────────────────────────────────────┐
│  Push: "What did you wear today?"                            │
│  → Opens WearLogScreen                                       │
│  → 1-tap confirm suggested outfit OR pick from closet        │
│  → Tag occasion → Log → Streak update → Celebration          │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌─── WEEKLY (Sunday 10:00 AM) ─────────────────────────────────┐
│  Push: "Your week in style"                                  │
│  → Opens WeeklyInsightsScreen                                │
│  → Utilization %, most worn, unworn nudge, color patterns    │
│  → Motivation to keep logging                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Retention Engine

### Streak Mechanics
- **Trigger**: Logging a wear (any method) = 1 streak day
- **Grace**: Streak maintained if user logged yesterday but not today
- **Milestones**: 3d → badge, 7d → "Style DNA" insight, 14d → pattern analysis, 30d → personalized color palette
- **At-risk nudge**: If streak > 0 and no log today → push at 8 PM

### Utilization Engine
- Tracks % of closet worn in last 30 days
- Unworn items surfaced as "Try something new" suggestions
- Weekly report shows utilization trend

### Feedback Loop
- Outfit ratings (1-5) feed back into `stylePreferenceStore`
- Color/pattern preferences evolve over time via `getLearnedColorPreferences()`
- `suggestionEngine` weights adapt to user behavior

### Nudge Hierarchy
1. **streak_at_risk** — highest priority (behavioral anchor)
2. **low_utilization** — if < 30% closet worn
3. **unworn_items** — if > 50% of items never worn
4. **none** — user is healthy

---

## 5. MVP Scope Definition

### MVP (Ship First)
8 screens. One behavioral loop end-to-end.

| Screen | Purpose |
|--------|---------|
| StyleQuiz | Onboarding — capture preference vector |
| Home | Dashboard — daily suggestion card + streak + quick actions |
| DailySuggestion | Morning entry — 3 scored outfit options |
| WearLog | Evening entry — 1-tap wear confirmation |
| MyCloset | Grid view of all items |
| ScanWardrobe + Camera + Review | Core input — video → AI → items |
| WeeklyInsights | Retention — stats, utilization, patterns |
| Profile | Settings, subscription, streak history |

### v2 (After Loop Works)
| Feature | Why v2 |
|---------|--------|
| AI Chat Stylist | Nice-to-have; core suggestions don't need it |
| Virtual Try-On | Expensive (Replicate API); premium-only |
| Trip Planner | Packing feature; depends on solid wardrobe data |
| Outfit Calendar | Scheduling; depends on wear log data |
| Occasion Outfits (Meeting, Date) | Contextual variations of core suggestion |
| Multi-language (RU, UZ) | Growth feature after product-market fit |

### Long-term (After Revenue)
| Feature | Why Later |
|---------|-----------|
| Flash Sales / Price Tracker | Different product (shopping vs wardrobe) |
| Avatar Creation | GPU-heavy, low retention impact |
| Magic Mirror | Tech demo, not core value |
| Design Room | Creative tool, niche audience |
| Social / Inspo Feed | Requires content pipeline + moderation |
| Brand Partnerships | Needs user scale first |

---

## 6. Data Architecture

### Canonical Types (single source of truth: `src/types/domain.ts`)

```
ClothingItem
  ├── id, userId, imageUrl, thumbnailUrl
  ├── category: 'top' | 'bottom' | 'shoes' | 'outerwear' | 'accessory'
  ├── subCategory, primaryColor, colorHex, pattern, material
  ├── brand?, name?, seasons[], occasions[]
  ├── wearCount, lastWornAt, isFavorite
  └── createdAt, updatedAt, detectionConfidence?

Outfit
  ├── id, userId, itemIds[]
  ├── occasion, generatedBy: 'ai' | 'user'
  ├── saved, wornCount, lastWornAt, rating?
  └── reasoning?, colorHarmony?, style?, createdAt

WearLog
  ├── id, userId, outfitId?, itemIds[]
  ├── date (YYYY-MM-DD), occasion?
  └── weatherTemp?, weatherCondition?, createdAt

DailySuggestion
  ├── outfit: Outfit
  ├── reason: string
  ├── weatherContext?: { temp, condition, city? }
  └── generatedAt

UserProfile
  ├── id, email, username, gender?
  ├── preferredStyles[], preferredColors[], bodyType?
  ├── tier: 'free' | 'premium', tierExpiresAt?
  └── onboardingComplete, lastActiveAt, streakDays
```

### Supabase Tables
```sql
clothing_items   — AI-detected + user-edited items
outfits          — generated + saved outfits with ratings
wear_logs        — behavioral tracking (items, occasion, weather)
users            — profile, preferences, subscription
daily_suggestions — cached per user per day (optional)
```

### Offline-First Sync
- All data cached in Zustand + AsyncStorage
- `pendingActions[]` queue for offline writes
- `syncToServer()` processes queue when online
- Wear logs capped at 500 locally, 200 persisted

---

## 7. COGS Model (Cost of Goods Sold per Active User)

| Cost Item | Per User/Month | Notes |
|-----------|---------------|-------|
| Supabase (DB + storage) | ~$0.02 | 100MB storage per user |
| AliceVision hosting | ~$0.50 | GPU inference, amortized |
| Weather API | ~$0.01 | Cached, 1 call/day |
| LLM calls (chat, explanations) | ~$1.00 | ~20 GPT-4 calls/month |
| Virtual try-on (Replicate) | ~$1.50 | ~15 generations/month (premium only) |
| Push notifications | ~$0.01 | Expo, minimal |
| **Total (Free user)** | **~$0.55** | No try-on, limited LLM |
| **Total (Premium user)** | **~$3.05** | All features |

### Unit Economics
- Premium price: $9.99/mo
- Premium COGS: ~$3.05/mo
- **Gross margin: ~69%**
- LTV (18 months avg): $9.99 × 18 × 0.69 = ~$124
- Target CAC: $10-15 (content marketing + referral)
- **LTV/CAC: 8-12x** (healthy for consumer subscription)

---

## 8. Growth Model (Realistic)

### Phase 1: Months 1-6 (Organic + Content)
- TikTok/Instagram Reels showing "60-second wardrobe scan"
- Target: 5,000 downloads, 500 active users, 50 premium ($500 MRR)
- CAC: ~$0 (organic content)

### Phase 2: Months 7-12 (Paid + Referral)
- Targeted ads (Instagram fashion audience)
- Referral program (invite friend → both get 1 week premium)
- Target: 20,000 downloads, 3,000 active, 300 premium ($3K MRR)
- CAC: ~$8

### Phase 3: Months 13-18 (Scale)
- Influencer partnerships
- App Store Optimization (ASO)
- Target: 50,000 downloads, 8,000 active, 800 premium ($8K MRR)
- CAC: ~$12

### Key Metrics to Track
- **D1/D7/D30 retention** — most critical
- **Scan completion rate** — onboarding funnel
- **Daily log rate** — behavioral loop health
- **Suggestions → wear log conversion** — core loop efficacy
- **Free → premium conversion** — monetization

---

## 9. Defensibility

### What's Defensible
1. **Vision pipeline** — SegFormer + region scanning + confidence filtering is real engineering. Competitors can replicate but it takes months of tuning.
2. **Behavioral data moat** — Every wear log makes suggestions better. After 30 days of logging, switching cost is high because a competitor doesn't know your habits.
3. **Wardrobe graph** — The scanned + tagged wardrobe is a personal dataset that compounds in value. Users won't re-scan at a competitor.

### What's NOT Defensible
1. LLM-powered chat (anyone can wrap GPT-4)
2. Virtual try-on (Replicate API is public)
3. Weather-based filtering (trivial to implement)

### Moat Strategy
- **Double down on: data network effects**. More logs → better suggestions → more logging → more data.
- **NOT on: AI model complexity**. Use off-the-shelf models efficiently, compete on UX and behavioral loop.

---

## 10. Scalability Path

### Current Architecture (MVP)
- Single Express.js server + single AliceVision instance
- Handles ~100 concurrent users
- Monthly cost: ~$50 (Render/Railway)

### Scale Path
1. **Separate scan from serve**: AliceVision runs as async job queue (Bull/Redis), not blocking API
2. **CDN for images**: Move cutout images to Cloudflare R2 or Supabase CDN
3. **Suggestion caching**: Pre-compute daily suggestions at 6 AM via cron, store in Supabase
4. **Edge functions**: Move weather + notification logic to Supabase Edge Functions
5. **Model optimization**: Quantize SegFormer for 2x inference speed, batch requests

---

## 11. File Structure (Post-Refactor)

```
src/
├── types/
│   └── domain.ts          ← SINGLE SOURCE OF TRUTH for all types
├── services/
│   ├── suggestionEngine.ts ← Rule-based outfit scoring (CORE)
│   ├── retentionService.ts ← Streaks, utilization, insights
│   ├── visionService.ts    ← AliceVision API client
│   ├── llmService.ts       ← GPT/Gemini text generation
│   ├── notificationService.ts ← Push scheduling
│   ├── aiService.ts        ← LEGACY (being decomposed)
│   └── ...
├── hooks/
├── theme/
│   └── tokens.ts           ← SINGLE SOURCE for design tokens
└── config/

store/
├── wardrobeStore.ts        ← Items, outfits, wearLogs, sync
├── stylePreferenceStore.ts ← Quiz results, feedback, learning
├── auth.ts                 ← Auth state, trial mode
├── subscriptionStore.ts    ← Tier, paywall
└── trialStore.ts

screens/ (MVP priority order)
├── StyleQuizScreen.tsx     ← 1. Onboarding
├── HomeScreen.tsx          ← 2. Dashboard
├── DailySuggestionScreen.tsx ← 3. Morning entry (NEW)
├── WearLogScreen.tsx       ← 4. Evening entry (NEW)
├── MyClosetScreen.tsx      ← 5. Wardrobe view
├── CameraScreen.tsx        ← 6. Scan input
├── WardrobeVideoScreen.tsx ← 6b. Video scan
├── ReviewScreen.tsx        ← 6c. Scan review
├── WeeklyInsightsScreen.tsx ← 7. Retention (NEW)
├── ProfileScreen.tsx       ← 8. Settings
└── ... (v2+ screens)

navigation/
├── types.ts               ← Navigation param types
├── RootNavigator.tsx       ← Auth → Onboarding → MVP → v2
└── TabNavigator.tsx        ← Bottom tabs
```
