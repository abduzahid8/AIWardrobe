
import { NavigatorScreenParams } from '@react-navigation/native';

export type ScannedItem = {
    id?: string;
    itemType?: string; // from ReviewScreen usage
    type?: string;
    color?: string;
    image?: string;
    imageUrl?: string;
    confidence?: number;
    style?: string;
    season?: string;
    description?: string;
};

export type ClothingItem = {
    id: number;
    image: string;
    x?: number;
    y?: number;
    name?: string;
    type?: string;
    gender?: string;
};

export type OutfitItem = {
    id?: string;
    items?: ScannedItem[];
};

export type RootStackParamList = {
    // Auth
    SignIn: undefined;
    SignUp: undefined;
    ForgotPassword: undefined;
    ResetPassword: undefined;

    // Main (tab container)
    Main: NavigatorScreenParams<TabParamList>;
    StyleQuiz: undefined;
    Paywall: undefined;

    // Wardrobe & Scanning
    AddOutfit: {
        date?: string;
        savedOutfits?: { [key: string]: ClothingItem[] };
    };
    ScanWardrobe: undefined;
    ReviewScan: { items: ScannedItem[] };
    WardrobeVideo: { videoUri?: string; imageUri?: string };
    ClothingStudio: undefined;
    Camera: undefined;
    ClothingDetailEditor: { imageUri?: string; detectedType?: string; detectedColor?: string; detectedItem?: any };

    // Creation & Design
    DesignRoom: undefined;
    NewOutfit: {
        selectedItems?: ClothingItem[];
        date?: string;
        savedOutfits?: { [key: string]: any[] };
    };

    // AI Features — names match registered Stack.Screen names exactly
    AIChat: { initialTab?: 'chat' | 'outfit' } | undefined;
    AIOutfit: undefined;
    AITryOn: undefined;
    AIHub: undefined;
    OutfitAI: { initialTab?: 'chat' | 'outfit' } | undefined;
    CreateAvatar: undefined;
    /** Gemini AI Stylist chat — accessible from any outfit/clothing context */
    StylistChat: { initialMessage?: string } | undefined;

    // Core Loop (MVP)
    DailySuggestion: undefined;
    WearLog: undefined;
    WeeklyInsights: undefined;

    // Calendar & Planning
    Calendar: undefined;
    TripPlanner: undefined;
    MeetingOutfit: undefined;

    // Shopping
    PriceTracker: undefined;
    FlashSales: undefined;
    FlashSaleEvent: { eventId: string };
    BusinessCasual: undefined;

    // Profile & Settings
    EmailOnboarding: undefined;
    OutfitDetail: { image?: string; outfit?: OutfitItem };
    MyCloset: undefined;
    ClothingDetail: { itemId: string; fullItem?: any };
    StyleGoals: undefined;
    WardrobeAnalytics: undefined;
    PrivacyPolicy: undefined;
    TermsOfService: undefined;
};

export type TabParamList = {
    Home: undefined;
    Closet: undefined;
    AI: undefined;
    Inspo: undefined;
    Profile: undefined;
};
