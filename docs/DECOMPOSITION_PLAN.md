# Screen Decomposition Plan

**Status:** Planned. Each screen below is to be migrated one-at-a-time with
tests and a manual smoke before merging.

**Rule of thumb:** a `screens/*` file should be ~200 lines of composition.
Anything bigger is extracted into `features/<name>/{hooks, components, utils, screens}`.

---

## Why

Four screens currently hold most of the app's logic in a single file each:

| Screen                     | LoC (approx) | Concerns fused in one file                                                |
|----------------------------|--------------|---------------------------------------------------------------------------|
| `AIOutfitmaker.tsx`        | ~2100        | Draft state, AI orchestration, collage rendering, save, share             |
| `ProfileScreen.tsx`        | ~2300        | Identity, saved looks, settings, language, logout, subscription, delete   |
| `HomeScreen.tsx`           | ~1700        | Weather, outfit generation, shop mix, nudges, collage mapping             |
| `MyClosetScreen.tsx`       | ~1200        | Grid, filters, item sheet, multi-select, sort, empty states               |

This makes code review, testing, and i18n-coverage audits impractical.
Decomposition is prerequisite to everything downstream: accessibility audit,
Sentry source-map tagging per feature, and code-splitting via Metro's
`unstable_allowRequireContext`.

---

## ProfileScreen → `features/profile/`

```
features/profile/
  screens/
    ProfileScreen.tsx                     # ~180 LoC composition only
  hooks/
    useProfileIdentity.ts                 # email, avatar upload, sign-out
    useSavedLooks.ts                      # fetch + paginate saved outfits
    useAccountDeletion.ts                 # wraps delete-account Edge Function
    useDesignTokens.ts                    # move from ProfileScreen lines 86-135
  components/
    ProfileHero.tsx                       # hero card + avatar picker
    SavedLooksGrid.tsx                    # the 2-column outfit grid
    LookDetailSheet.tsx                   # modal bottom sheet for a look
    SettingsList.tsx                      # language, theme, notifications
    SubscriptionRow.tsx                   # tier badge + manage link
    LogoutRow.tsx
    DeleteAccountRow.tsx                  # required by Apple 5.1.1(v)
  utils/
    formatters.ts                         # titleCase, getErrorMessage
```

**Migration steps:**

1. Copy `useDesignTokens` verbatim into `features/profile/hooks/useDesignTokens.ts`. Update all imports.
2. Extract `SavedOutfit` interface → `features/profile/types.ts`.
3. Move the hero (lines ~300-500) into `ProfileHero.tsx`. Props: `{ user, onEditAvatar }`.
4. Move saved-looks grid + its query into `useSavedLooks` + `SavedLooksGrid`.
5. Move the three settings groups (account / appearance / legal) into `SettingsList`, each row a declarative entry.
6. Replace the logout + delete inline handlers with dedicated rows driven by `useAccountDeletion`.
7. `ProfileScreen.tsx` becomes: `<Hero /> <SavedLooksGrid /> <SettingsList /> <SubscriptionRow /> <LogoutRow /> <DeleteAccountRow />`.

**Tests to add before merge:**

- `useAccountDeletion.test.ts`: happy path + Edge Function error → user-visible alert.
- `useSavedLooks.test.ts`: pagination, empty state, RLS error.
- Snapshot test on `ProfileHero` to guard the identity UI.

---

## HomeScreen → `features/home/`

```
features/home/
  screens/
    HomeScreen.tsx                        # ~200 LoC composition
  hooks/
    useWeatherContext.ts                  # location → weather edge fn
    useDailyOutfit.ts                     # AI outfit + fallback combos
    useShopCatalogSegments.ts             # memoized tops/bottoms/shoes/outerwear
    useWardrobeEssentials.ts              # essentials catalog mix
  components/
    HomeHeader.tsx                        # already exists — keep
    WeatherWidget.tsx                     # MOVE from components/home/
    TodaysLookCard.tsx                    # hero video + CTA
    OutfitSuggestionCarousel.tsx          # horizontal outfit list
    EssentialsSection.tsx                 # shop-essentials grid
    NudgeCard.tsx                         # the activePrompt card
  utils/
    collageMappers.ts                     # mapLegacyOutfitItemsForCollage +
                                          # mapAiOutfitItemsForCollage
    outerwearHeuristics.ts                # isOuterwearItem
```

**Notes:** the weather fetch has already been migrated to the `weather` Edge
Function. The collage mapper duplication that currently sits mid-file will
collapse to a single pure utility once moved.

---

## AIOutfitmaker → `features/outfit-ai/`

```
features/outfit-ai/
  screens/
    AIOutfitmakerScreen.tsx               # ~250 LoC
  hooks/
    useOutfitDraft.ts                     # draft CRUD, items, undo
    useOutfitGeneration.ts                # call Edge Function, poll
    useOutfitSave.ts                      # persist to wardrobe
  components/
    DraftCanvas.tsx                       # the main drag/tap canvas
    AIPromptBar.tsx                       # prompt input + model picker
    ItemPickerSheet.tsx                   # bottom sheet for wardrobe items
    OutfitActionsBar.tsx                  # save/share/like
  utils/
    outfitSerialization.ts
    colorPalette.ts
```

**Risk:** this screen has the most inter-dependent state. Do it last,
after Home and Profile are done so the migration pattern is proven.

---

## MyClosetScreen → `features/wardrobe/`

`features/wardrobe/` already exists. Migrate the remaining screen logic into it.

```
features/wardrobe/
  screens/
    MyClosetScreen.tsx                    # ~180 LoC
  hooks/
    useClosetFilters.ts                   # category / color / season
    useMultiSelect.ts                     # bulk delete, bulk move
  components/
    ClosetGrid.tsx                        # virtualized grid
    FilterBar.tsx
    ItemActionSheet.tsx
    EmptyClosetState.tsx
```

---

## Execution order

1. **ProfileScreen** (not currently being edited) — use it to prove the pattern.
2. **MyClosetScreen** (stable) — reuses the same pattern.
3. **AIOutfitmaker** — larger surface, but stable.
4. **HomeScreen** — do last. It is actively being edited as of 2026-04-24.

## Acceptance checklist per screen

- [ ] `git diff --stat` shows the original `screens/*.tsx` shrunk to <250 LoC.
- [ ] All tests green: `npm test && npx tsc --noEmit`.
- [ ] Manual smoke on both iOS and Android.
- [ ] `@/Users/<me>/Desktop/AIWardrobe/src/.windsurfrules` grep for the old symbol shows zero hits.
- [ ] Sentry source-map tag includes the new `features/<name>` path prefix.
