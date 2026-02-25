
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
    Camera: undefined;

    // Creation & Design
    DesignRoom: undefined;
    NewOutfit: {
        selectedItems?: ClothingItem[];
        date?: string;
        savedOutfits?: { [key: string]: any[] };
    };

    // AI Features — names match registered Stack.Screen names exactly
    AIChat: undefined;
    AIOutfit: undefined;
    AITryOn: undefined;
    AIHub: undefined;
    OutfitAI: undefined;
    CreateAvatar: undefined;

    // Calendar & Planning
    Calendar: undefined;
    TripPlanner: undefined;
    MeetingOutfit: undefined;

    // Shopping
    PriceTracker: undefined;
    FlashSales: undefined;
    FlashSaleEvent: { eventId: string };

    // Profile & Settings
    EmailOnboarding: undefined;
    OutfitDetail: { image?: string; outfit?: OutfitItem };
    MyCloset: undefined;
    StyleGoals: undefined;
};

export type TabParamList = {
    Home: undefined;
    Closet: undefined;
    AI: undefined;
    Inspo: undefined;
    Profile: undefined;
};
