# Maestro E2E Testing Guide

## Setup

```bash
# 1. Start Metro bundler
npx expo start --port 8081

# 2. Launch app on booted simulator
xcrun simctl launch booted org.name.AIWardrobe

# 3. Wait for app to fully render (~10s)
# 4. Run tests
cd .maestro
../scripts/run_maestro_tests.sh --no-build
```

Or use the full runner (builds + runs):
```bash
./scripts/run_maestro_tests.sh
```

## Working Flows

| Flow | What it does | Notes |
|------|-------------|-------|
| `nav_home.yaml` | Launch + tap Home tab | Requires app running or will hang |
| `nav_closet.yaml` | Launch + tap Closet tab | Same |
| `nav_inspo.yaml` | Launch + tap Inspo tab | Same |
| `nav_profile.yaml` | Launch + tap Profile tab | Same |
| `nav_closet_inspo.yaml` | Home → Closet → Inspo | 2 taps in one flow |
| `nav_3tabs.yaml` | Home → Closet → Inspo → Profile | 3 taps |
| `nav_4tabs_nolaunch.yaml` | All 4 tabs, no launchApp | Preferred - app must be pre-launched |
| `nav_4tabs_screenshots.yaml` | All 4 tabs + screenshots | Captures each tab state |

## How It Works

### The iOS 26 + React Native Bug

iOS 26.1's Accessibility Server returns **invalid element frames** (`kAXErrorInvalidUIElement`) for React Native 0.81.5 views. This crashes Maestro's `viewHierarchy` queries. Root cause: `kAXErrorInvalidUIElement` → zero/1x1 frames.

### Workaround: Coordinate Taps

All interactions use percentage-based coordinates:

```yaml
- tapOn:
    point: "37%,94%"
```

Tab positions on iPhone 17 Pro (402×874pt):
| Tab | X% | Y% |
|-----|----|----|
| Home | 12% | 94% |
| Closet | 37% | 94% |
| Inspo | 62% | 94% |
| Profile | 87% | 94% |

### Stability Limit

Approximately **5 `waitForAnimationToEnd` calls per flow** before the accessibility server corrupts. Beyond that, simulator needs a full erase + reboot.

### Key Rule: Pre-launch the App

`launchApp` in Maestro calls `XCUIApplication().launch()`, which queries the accessibility hierarchy and **hangs** when the app isn't running. To avoid this:

1. Launch the app manually: `xcrun simctl launch booted org.name.AIWardrobe`
2. Use flows **without** `launchApp` (e.g., `nav_4tabs_nolaunch.yaml`)
3. Or use flows WITH `launchApp` but only after the app is already running

## What Can't Be Tested

- **Text-based element matching** (`tapOn: { text: "..." }`, `assertVisible`): crashes with `kAXErrorInvalidUIElement`
- **`launchApp` on cold start**: hangs the XCUITest driver
- **More than ~5 hierarchy queries per flow**: accessibility server corruption
- **Scrolling**: requires element detection to know when scroll completes

## Quick Start

```bash
# Terminal 1: Start Metro
npx expo start --port 8081

# Wait for "Waiting on http://localhost:8081"
# Terminal 2: Launch app
xcrun simctl launch booted org.name.AIWardrobe

# Wait 10s for app to render
# Terminal 3: Run tests
cd .maestro
maestro test nav_4tabs_screenshots.yaml --no-reinstall-driver
```

## Tab Coordinate Verification

If the app layout changes (e.g., different device, tab bar redesign), re-verify coordinates:
1. Take a screenshot: `xcrun simctl io booted screenshot screenshot.png`
2. Open in Preview, measure the tab bar area
3. Update coordinates in the flows
