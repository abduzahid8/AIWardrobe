# Alta-Style Screen Templates

This folder contains screen templates that closely match Alta Daily's design patterns.
Use these as starting points and customize for your AIWardrobe brand.

## Templates Included

| Screen | File | Description |
|--------|------|-------------|
| **Landing** | `AltaLandingScreen.tsx` | Marketing/onboarding page with hero, features, style goals |
| **Home** | `AltaHomeScreen.tsx` | Wardrobe grid with weather widget, progress toast, bottom nav |
| **Chat** | `AltaChatScreen.tsx` | AI stylist interface with suggestions, chat bubbles |
| **Profile** | `AltaProfileScreen.tsx` | User profile with avatar, looks/trips tabs, friends card |
| **Shop** | `AltaShopScreen.tsx` | Product grid with categories, like buttons |

## Color System

All templates use the Alta monochromatic palette defined in `AltaColors.ts`:

```typescript
const ALTA = {
    bg: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#000000',
    textSecondary: '#666666',
    textMuted: '#8E8E8E',
    border: '#E5E5E5',
};
```

## Design Patterns Used

1. **Monochromatic** - Black/white/grays with products providing color
2. **Tactile Press** - Scale to 0.97 on touch with haptic feedback
3. **Weather Context** - Temperature in header for outfit relevance
4. **Progress Gamification** - "Add X items to unlock..." toast
5. **Pill Buttons** - Rounded buttons for avatar, categories
6. **Clean Typography** - Sans-serif, proper hierarchy
7. **Subtle Shadows** - Instead of borders

## How to Use

1. Copy the screen you want to your main `/screens` folder
2. Rename it (e.g., `AltaHomeScreen.tsx` → `HomeScreen.tsx`)
3. Update imports and navigation
4. Customize colors, text, and functionality
5. Replace placeholder data with your actual data

## Example: Replace HomeScreen

```typescript
// In TabNavigator.tsx
import AltaHomeScreen from '../screens/templates/AltaHomeScreen';

// Replace:
<Tab.Screen name="Home" component={HomeScreen} />
// With:
<Tab.Screen name="Home" component={AltaHomeScreen} />
```
