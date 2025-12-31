// Navigation types for the app

export type RootStackParamList = {
    // Auth screens
    SignIn: undefined;
    SignUp: undefined;

    // Main screens
    Home: undefined;
    Tabs: undefined;

    // Wardrobe screens
    AddOutfit: {
        date?: string;
        savedOutfits?: { [key: string]: any[] };
    };
    DesignRoom: {
        selectedItems: ClothingItem[];
        date: string;
        savedOutfits?: { [key: string]: any[] };
    };
    NewOutfit: {
        selectedItems: ClothingItem[];
        date: string;
        savedOutfits?: { [key: string]: any[] };
    };

    // AI screens
    AIChat: undefined;
    AIAssistant: { initialMessage?: string };
    AIOutfit: undefined;
    AIOutfitmaker: undefined;
    AITryOn: undefined;
    ScanWardrobe: undefined;
    WardrobeVideo: undefined;
    WardrobeStats: undefined;
};

export type TabParamList = {
    Home: undefined;
    Discover: undefined;
    Wardrobe: undefined;
    Profile: undefined;
    Add: undefined; // Keeping for backward compatibility if needed
};

export type ClothingItem = {
    id: number;
    image: string;
    name?: string;
    type?: string;
    gender?: string;
};
