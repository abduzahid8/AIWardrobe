/**
 * Navigation Types
 * Types for React Navigation
 */

import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { DetectedItem } from './api';

/**
 * Root stack parameter list
 */
export type RootStackParamList = {
    // Auth screens
    SignIn: undefined;
    SignUp: undefined;

    // Main screens (tabs)
    Home: undefined;
    MainTabs: undefined;

    // Feature screens
    AddOutfit: undefined;
    AIChat: undefined;
    AIOutfit: undefined;
    AITryOn: undefined;
    ScanWardrobe: undefined;
    DesignRoom: undefined;
    NewOutfit: undefined;
    WardrobeVideo: undefined;

    // Review screen with params
    ReviewScan: {
        items: DetectedItem[];
    };

    // Clothing detail
    ClothingDetail: {
        itemId: string;
    };

    // Outfit detail
    OutfitDetail: {
        outfitId: string;
    };
};

/**
 * Tab navigator parameter list
 */
export type TabParamList = {
    HomeTab: undefined;
    DiscoverTab: undefined;
    AddTab: undefined;
    ProfileTab: undefined;
};

/**
 * Navigation prop type helper
 */
export type NavigationProp<T extends keyof RootStackParamList> =
    NativeStackNavigationProp<RootStackParamList, T>;

/**
 * Route prop type helper
 */
export type RouteProps<T extends keyof RootStackParamList> =
    RouteProp<RootStackParamList, T>;

/**
 * Screen props type helper - combines navigation and route
 */
export type ScreenProps<T extends keyof RootStackParamList> = {
    navigation: NavigationProp<T>;
    route: RouteProps<T>;
};
