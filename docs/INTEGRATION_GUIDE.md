# AIWardrobe — Integration Guide

This document describes how all rebuilt services and components connect to each other.

---

## Architecture Overview

```
App.tsx
└── TabNavigator (Home | Closet | Chat | Inspo | Profile)
    ├── HomeScreen
    │   ├── OutfitSuggestionCarousel
    │   │   └── OutfitCard (save → wardrobeStore.addOutfit)
    │   ├── OutfitCarouselSkeleton (loading state)
    │   └── Upload queue status banner
    ├── ChatScreen  ← Gemini with full wardrobe context
    ├── InspoScreen ← "From Your Closet" via generateVarietyOutfits
    └── ProfileScreen → WardrobeAnalyticsScreen
                     → AITryOnScreen (push, not tab)
```

---

## Service Layer

### `src/services/suggestionEngine.ts`
Core outfit logic. Call order:

| Function | Used by | Description |
|---|---|---|
| `generateDailyOutfits` | HomeScreen | 4 variants (work / smart casual / casual / active) |
| `generateVarietyOutfits` | InspoScreen | 6 maximally-diverse outfits |
| `generateOutfitsForItem` | HomeScreen (unworn row CTA) | Anchor-piece mode |
| `generateSuggestions` | Any | Raw scored outfit list |

All functions return `ScoredOutfit[]`. They are **purely synchronous** — no network calls.

### `src/services/aiProviderService.ts`
Single singleton `aiProvider`. Key methods:

```ts
// Chat with full context injection
const result: ChatResult = await aiProvider.chat(message, {
    items,
    wearLogs,
    weather,          // optional
    stylePreferences, // optional
    dislikedOutfitSummaries: useWardrobeStore.getState().getDislikedSummaries(),
}, onSlowResponseCallback);

// Analyze a clothing photo
const result: AnalyzeImageResult = await aiProvider.analyzeImage(base64);

// Process full upload pipeline
const result: ProcessUploadResult = await aiProvider.processUpload(base64);
```

Context is cached for **15 minutes**. Call `invalidateContextCache()` after adding/removing items.

### `src/services/uploadQueue.ts`
Singleton `uploadQueue`. Initialize once in `App.tsx`:

```ts
// App.tsx — app startup
uploadQueue.init();
uploadQueue.setSuccessHandler(async (tempId, result) => {
    await useWardrobeStore.getState().addItem({ ... });
    invalidateContextCache();
});

// App.tsx — logout
uploadQueue.destroy();
```

Subscribe in any component for live banner updates:
```ts
useEffect(() => {
    return uploadQueue.subscribe(setQueueState);
}, []);
```

---

## Store Layer

### `store/wardrobeStore.ts`
All wardrobe data. Key additions:

```ts
// Disliked outfits — persisted across sessions
dislikeOutfit(itemIds)        // call from OutfitCard "dislike" button
undislikeOutfit(itemIds)      // undo
getDislikedSummaries()        // → string[] for AI context
```

Pass disliked summaries to `aiProvider.chat` so Gemini never repeats them.

### `store/trialStore.ts`
Paywall gate. The `pendingPaywall` flag fires 800 ms after the 5th AI use:

```ts
// In any screen that triggers AI
incrementTrialCount();

// In RootNavigator or a shared component — watch for bottom sheet
const { pendingPaywall, dismissPaywall } = useTrialStore();
useEffect(() => {
    if (pendingPaywall) {
        // show bottom sheet
        dismissPaywall(); // call once sheet is visible
    }
}, [pendingPaywall]);
```

---

## Component Layer

### `src/components/OutfitCard.tsx`
Props wiring:

```ts
<OutfitCard
    outfit={scoredOutfit}
    allItems={wardrobeItems}
    onSave={(outfitId, itemIds) => wardrobeStore.addOutfit(...)}
    onDislike={(itemIds) => wardrobeStore.dislikeOutfit(itemIds)}
    onWear={(itemIds) => wardrobeStore.logWear(itemIds)}
    onAvatarCreate={(itemIds) => navigation.navigate('AITryOn', { itemIds })}
/>
```

### `src/components/OutfitSuggestionCarousel.tsx`
Wraps `OutfitCard` in a full-width `ScrollView`. Auto-advances to next outfit on dislike.

### `src/components/SkeletonLoader.tsx`
Exported variants:

| Export | Use |
|---|---|
| `OutfitCarouselSkeleton` | HomeScreen loading |
| `ClosetGridSkeleton` | MyClosetScreen loading |
| `ChatBubbleSkeleton` | ChatScreen initial load |
| `AnalyticsSkeleton` | WardrobeAnalyticsScreen |
| `InspoGridSkeleton` | InspoScreen |

### `src/components/AvatarDisplay.tsx`
Renders layered clothing on a silhouette. Used by AITryOnScreen and outfit preview modals:

```ts
<AvatarDisplay
    items={selectedItems}
    size={280}
    showSilhouette
/>
```

---

## Navigation

`TabNavigator` routes: `Home | Closet | Chat | Inspo | Profile`

`AITryOnScreen` is a **push screen** (not a tab). Navigate to it:
```ts
navigation.navigate('AITryOn', { itemIds: [...] });
```

Register it in `RootNavigator.tsx` as a stack screen above the tab navigator.

---

## Environment Variables

See `api/.env.example` and `src/config/env.ts`. Required:

| Variable | Description |
|---|---|
| `EXPO_PUBLIC_API_URL` | Node.js API server URL |
| `EXPO_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `EXPO_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon key |
| `EXPO_PUBLIC_WEATHER_API_KEY` | OpenWeatherMap key (HomeScreen) |
| `EXPO_PUBLIC_GEMINI_API_KEY` | Google Gemini key (aiProviderService) |

---

## Known Patterns

- **No circular imports**: `aiProviderService` lazy-requires `wardrobeStore` via `require()`.
- **Offline-first**: `uploadQueue` and `wardrobeStore.pendingActions` both survive app restarts.
- **Context cache**: Invalidate on every item add/remove — call `invalidateContextCache()`.
- **Disliked outfits**: `dislikedOutfitKeys` is stored sorted so the same combo matches regardless of order.
