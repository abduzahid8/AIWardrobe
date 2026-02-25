/**
 * Centralized route name constants.
 * Use these instead of raw strings to prevent typos and enable refactoring.
 */

export const ROUTES = {
    // Auth
    SIGN_IN: 'SignIn',
    SIGN_UP: 'SignUp',

    // Main (tab container)
    MAIN: 'Main',
    STYLE_QUIZ: 'StyleQuiz',
    PAYWALL: 'Paywall',

    // Wardrobe & Scanning
    ADD_OUTFIT: 'AddOutfit',
    SCAN_WARDROBE: 'ScanWardrobe',
    REVIEW_SCAN: 'ReviewScan',
    WARDROBE_VIDEO: 'WardrobeVideo',
    CAMERA: 'Camera',

    // Creation & Design
    DESIGN_ROOM: 'DesignRoom',
    NEW_OUTFIT: 'NewOutfit',

    // AI Features
    AI_CHAT: 'AIChat',
    AI_OUTFIT: 'AIOutfit',
    AI_TRY_ON: 'AITryOn',
    AI_HUB: 'AIHub',
    OUTFIT_AI: 'OutfitAI',
    CREATE_AVATAR: 'CreateAvatar',

    // Calendar & Planning
    CALENDAR: 'Calendar',
    TRIP_PLANNER: 'TripPlanner',
    MEETING_OUTFIT: 'MeetingOutfit',

    // Shopping
    PRICE_TRACKER: 'PriceTracker',
    FLASH_SALES: 'FlashSales',
    FLASH_SALE_EVENT: 'FlashSaleEvent',

    // Profile & Settings
    EMAIL_ONBOARDING: 'EmailOnboarding',
    OUTFIT_DETAIL: 'OutfitDetail',
    MY_CLOSET: 'MyCloset',
    STYLE_GOALS: 'StyleGoals',
} as const;

// Tab names
export const TABS = {
    HOME: 'Home',
    CLOSET: 'Closet',
    AI: 'AI',
    INSPO: 'Inspo',
    PROFILE: 'Profile',
} as const;

export type RouteName = (typeof ROUTES)[keyof typeof ROUTES];
export type TabName = (typeof TABS)[keyof typeof TABS];
