
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
    // First-run — language-only gate, shown before anything else
    LanguageSelection: undefined;

    // Auth
    SignIn: undefined;
    SignUp: undefined;
    ForgotPassword: undefined;
    ResetPassword: undefined;
    AuthCallback: undefined;

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
    ClothingDetailEditor: { scanReviewId?: string; scanReviewItemId?: string; imageUri?: string; detectedType?: string; detectedColor?: string; detectedItem?: any; detectedStyle?: string; detectedMaterial?: string; aiConfidence?: number; detectedDescription?: string; existingItem?: any };

    // AI Features — names match registered Stack.Screen names exactly
    AIOutfit: {
        source?: 'wardrobe' | 'shop';
        calendarDate?: string;
        initialStyle?: string;
        /** If set, AI generates outfits that always include this wardrobe item as the anchor. */
        baseItemId?: string;
        /** Optional preview of the anchor item shown immediately while the AI loads. */
        baseItem?: { id: string; imageUrl?: string; name?: string; type?: string; macroCategory?: string; color?: string };
    } | undefined;
    OutfitInspo: undefined;
    CreateAvatar: undefined;

    // Calendar
    Calendar: { initialDate?: string } | undefined;

    // Profile & Settings
    MyCloset: undefined;
    ClothingDetail: { itemId: string; fullItem?: any };
    PrivacyPolicy: undefined;
    TermsOfService: undefined;
    AdminPanel: undefined;
    Guide: undefined;
    /** Body-fit profile (Month 1 of body-fit plan). Editable in Profile tab. */
    BodyProfile: undefined;
};

export type TabParamList = {
    Home: undefined;
    Closet: undefined;
    AI: undefined;
    Inspo: { focusShopItemId?: string; focusShopItemName?: string } | undefined;
    Profile: undefined;
};
