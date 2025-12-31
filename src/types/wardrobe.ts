// ============================================
// CLOTHING ITEM TYPES
// ============================================

export type ClothingCategory =
    | 'tops'
    | 'bottoms'
    | 'dresses'
    | 'outerwear'
    | 'shoes'
    | 'accessories'
    | 'bags';

export type ClothingStyle =
    | 'casual'
    | 'formal'
    | 'sport'
    | 'streetwear'
    | 'business'
    | 'evening';

export interface ClothingItem {
    id: string;
    uniqueId?: string;
    type: string;
    category: ClothingCategory;
    color: string;
    colors?: string[];
    style: ClothingStyle;
    material?: string;
    brand?: string;
    image: string;
    imageUrl?: string;
    description?: string;
    tags?: string[];
    isFavorite?: boolean;
    isSaved?: boolean;
    createdAt?: string;
    updatedAt?: string;
    wearCount?: number;
    lastWorn?: string;
}

// ============================================
// OUTFIT TYPES
// ============================================

export interface Outfit {
    _id: string;
    id?: string;
    items: ClothingItem[];
    occasion: string;
    style?: ClothingStyle;
    date?: string;
    weather?: string;
    temperature?: number;
    notes?: string;
    isFavorite?: boolean;
    createdAt: string;
    updatedAt?: string;
}

export interface OutfitSuggestion {
    id: string;
    items: ClothingItem[];
    occasion: string;
    confidence: number;
    reasoning?: string;
    weatherAppropriate?: boolean;
}

// ============================================
// AI ANALYSIS TYPES
// ============================================

export interface AIAnalysisResult {
    itemType: string;
    color: string;
    style: ClothingStyle;
    material?: string;
    description?: string;
    confidence?: number;
}

export interface VideoAnalysisResult {
    success: boolean;
    detectedItems: AIAnalysisResult[];
    bestFrameIndex?: number;
    processingTimeMs?: number;
}

export interface ProductPhotoResult {
    success: boolean;
    imageUrl: string;
    analysis?: {
        colors: string[];
        primaryColor: string;
        pattern?: string;
        material?: string;
        confidence: number;
    };
    quality?: {
        overall: number;
        ecommerceReady: boolean;
        issues: string[];
    };
}

// ============================================
// USER TYPES
// ============================================

export interface User {
    _id: string;
    username: string;
    email: string;
    profileImage?: string;
    followers?: string[];
    following?: string[];
    preferences?: UserPreferences;
    createdAt?: string;
}

export interface UserPreferences {
    favoriteStyles?: ClothingStyle[];
    favoriteColors?: string[];
    sizes?: {
        tops?: string;
        bottoms?: string;
        shoes?: string;
    };
    designers?: string[];
}

// ============================================
// API RESPONSE TYPES
// ============================================

export interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
    hasMore: boolean;
}

// ============================================
// NAVIGATION TYPES
// ============================================

export type RootStackParamList = {
    Home: undefined;
    SignIn: undefined;
    SignUp: undefined;
    AddOutfit: undefined;
    AIChat: { initialMessage?: string };
    AIOutfit: undefined;
    AITryOn: undefined;
    ScanWardrobe: undefined;
    ReviewScan: { items: AIAnalysisResult[] };
    DesignRoom: undefined;
    NewOutfit: undefined;
    WardrobeVideo: undefined;
    Calendar: undefined;
    Profile: undefined;
};

// ============================================
// CALENDAR TYPES
// ============================================

export interface CalendarEvent {
    id: string;
    date: string;
    outfit?: Outfit;
    occasion?: string;
    notes?: string;
}

export interface WeatherData {
    date: string;
    temperature: number;
    condition: 'sunny' | 'cloudy' | 'rainy' | 'snowy' | 'windy';
    humidity?: number;
}

// ============================================
// STORE TYPES
// ============================================

export interface AuthState {
    isAuthenticated: boolean;
    isTrialMode: boolean;
    user: User | null;
    token: string | null;
}

export interface WardrobeState {
    items: ClothingItem[];
    outfits: Outfit[];
    favorites: string[];
    isLoading: boolean;
}
