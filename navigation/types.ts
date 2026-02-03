
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

    // Main
    Home: NavigatorScreenParams<TabParamList>;

    // Wardrobe & Scanning
    AddOutfit: {
        date?: string;
        savedOutfits?: { [key: string]: ClothingItem[] };
    };
    ScanWardrobe: undefined;
    ReviewScan: { items: ScannedItem[] }; // Confirmed usage
    WardrobeVideo: { videoUri?: string; imageUri?: string };
    Camera: undefined;

    // Creation & Design
    DesignRoom: undefined; // DesignRoomScreen.tsx doesn't seem to access params via route
    NewOutfit: {
        selectedItems?: ClothingItem[];
        date?: string;
        savedOutfits?: { [key: string]: any[] }; // from NewOutfitScreen.tsx
    };

    // AI Features
    AIChat: undefined;
    AIAssistant: { initialMessage?: string };
    AIOutfit: undefined;
    AIOutfitmaker: undefined;
    AITryOn: undefined;
    AIHub: undefined;
    OutfitAI: undefined;

    // Others
    Calendar: undefined;
    EmailOnboarding: undefined;
    TripPlanner: undefined;
    OutfitDetail: { image?: string; outfit?: OutfitItem };
    Paywall: undefined;
    MeetingOutfit: undefined;
    PriceTracker: undefined;
    FlashSales: undefined;
    FlashSaleEvent: { eventId: string };
    MyCloset: undefined;
    StyleGoals: undefined;
};

export type TabParamList = {
    Home: undefined;
    Closet: undefined; // Changed from Wardrobe in original types to match TabNavigator usage
    AI: undefined;
    Inspo: undefined;
    Profile: undefined;
    Discover: undefined;
    Add: undefined;
    Wardrobe: undefined;
};
