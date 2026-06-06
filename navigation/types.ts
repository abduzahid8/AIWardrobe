
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

/**
 * Layout-positioned clothing item for canvas/collage views.
 * NOT the same as the domain ClothingItem (src/types/domain.ts).
 */
export type PositionedClothingItem = {
    id: number;
    image: string;
    x?: number;
    y?: number;
    name?: string;
    type?: string;
    gender?: string;
};

/** @deprecated Use PositionedClothingItem instead */
export type ClothingItem = PositionedClothingItem;

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
    TrialExpired: undefined;
    PromoCode: undefined;

    // Wardrobe & Scanning
    ScanWardrobe: undefined;
    ReviewScan: { items: ScannedItem[] };
    WardrobeVideo: { videoUri?: string; imageUri?: string };
    Camera: undefined;
    ClothingDetailEditor: { imageUri?: string; detectedType?: string; detectedColor?: string; detectedItem?: any; detectedStyle?: string; detectedMaterial?: string; aiConfidence?: number; detectedDescription?: string; existingItem?: any };

    // AI Features — names match registered Stack.Screen names exactly
    AIChat: { initialTab?: 'chat' | 'outfit' } | undefined;
    AIOutfit: {
        source?: 'wardrobe' | 'shop';
        calendarDate?: string;
        initialStyle?: string;
        /** If set, AI generates outfits that always include this wardrobe item as the anchor. */
        baseItemId?: string;
        /** Optional preview of the anchor item shown immediately while the AI loads. */
        baseItem?: { id: string; imageUrl?: string; name?: string; type?: string; macroCategory?: string; color?: string };
    } | undefined;
    AITryOn: undefined;
    OutfitInspo: undefined;
    OutfitAI: { initialTab?: 'chat' | 'outfit' } | undefined;
    CreateAvatar: undefined;
    /** Gemini AI Stylist chat — accessible from any outfit/clothing context */
    StylistChat: { initialMessage?: string } | undefined;

    // Calendar
    Calendar: undefined;

    // Profile & Settings
    MyCloset: undefined;
    ClothingDetail: { itemId: string; fullItem?: any };
    WardrobeAnalytics: undefined;
    WeeklyInsights: undefined;
    FlashSaleEvent: { eventId: string };
    PrivacyPolicy: undefined;
    TermsOfService: undefined;
    AdminPanel: undefined;
    Guide: undefined;
};

export type TabParamList = {
    Home: undefined;
    Closet: undefined;
    AI: undefined;
    Inspo: undefined;
    Profile: undefined;
};
