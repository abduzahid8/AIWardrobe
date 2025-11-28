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
    AIOutfit: undefined;
    AITryOn: undefined;
    ScanWardrobe: undefined;
};

export type TabParamList = {
    Home: undefined;
    Add: undefined;
    Profile: undefined;
};

export type ClothingItem = {
    id: number;
    image: string;
    name?: string;
    type?: string;
    gender?: string;
};
