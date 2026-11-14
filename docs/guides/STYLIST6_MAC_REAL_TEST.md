# Stylist 6 — manual testing on Mac, isolated PR #15 worktree

Runs an isolated development copy, without touching the existing main
checkout and without merging/promoting production. New footwear is NOT
published. The style-picker is development-only. Cloud staging is pinned
separately and may not yet include this branch's changes.

## Install update in a second worktree

On your Mac, with Node 22, Xcode and the original AIWardrobe checkout:

    cd ~/AIWardrobe
    git fetch origin codex/stylist6-visual-first-20261110
    git worktree add --detach ../AIWardrobe-stylist6-test origin/codex/stylist6-visual-first-20261110
    cd ../AIWardrobe-stylist6-test
    git rev-parse --short HEAD

Reuse existing configuration locally, never share secrets in chat:

    test -f ~/AIWardrobe/.env || { echo "Original .env missing"; exit 1; }
    ln -s ~/AIWardrobe/.env .env
    npm ci
    npm run typecheck
    node --test scripts/stylist6/live_footwear_replay.test.mjs

## Run true live-catalog (read-only) comparison locally

    mkdir -p .private/stylist6-live
    node scripts/stylist6/frozen_catalog.mjs .private/stylist6-live/catalog.json --per-slot 96
    node scripts/stylist6/live_footwear_replay.mjs \
      --catalog .private/stylist6-live/catalog.json \
      --out .private/stylist6-live/comparison \
      --exclude-brands ECCO

This reads only public catalog tables via Supabase anon RLS and runs
deterministic outfit retrieval without image downloads, paid NVIDIA or
quota debit. The .private folder is git-ignored. Share only the concise
summary.json metrics, NEVER raw catalog, .env, HMAC or tokens.

## Open the iOS development client

On the same Mac, with an iOS simulator configured:

    cd ~/AIWardrobe-stylist6-test
    NVIDIA_API_KEY= \
    EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true \
    EXPO_PUBLIC_STYLIST_DIAGNOSTICS=true \
    npm run dev:stylist6

This starts the local Node Stylist 6 bridge and Metro, and asks to open
the simulator. The empty NVIDIA_API_KEY override means this local process
cannot submit paid NVIDIA calls even if the dev checkbox is toggled;
the cloud staging endpoint is a separate system and is not opted into.
If Expo requires a dev build, in this isolated worktree run
npx expo run:ios once, then retry the startup command.

In AI Outfit Maker choose Stylist 6 (local).
**Switch NVIDIA visual review OFF before Generate.** It can be ON by
default in this developer picker and may spend credits. Do not select
Stylist 6 (cloud) until cloud staging is repinned and quota approved.

Manually inspect three scenarios: Old Money/Work, Classic/Date Night,
Smart Casual/Everyday. Check three wearable complete looks, actual
distinctness and footwear, image display, native currency and budget,
and Buy links opening the correct seller. Screenshots are for private
review; provide only redacted diagnostics and model-version text.

## Refresh this isolated test worktree later

    cd ~/AIWardrobe
    git fetch origin codex/stylist6-visual-first-20261110
    cd ~/AIWardrobe-stylist6-test
    git status --short
    git switch --detach origin/codex/stylist6-visual-first-20261110
    npm ci

Only switch when git status is clean. Ctrl-C stops Metro/bridge.
To delete when clean, from ~/AIWardrobe run:
    git worktree remove ../AIWardrobe-stylist6-test
