/**
 * API Response Types
 * Common types for API responses and requests
 */

/**
 * Generic API response wrapper
 */
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}

/**
 * Detected clothing item from AI analysis
 */
export interface DetectedItem {
    itemType: string;
    color: string;
    style: ClothingStyle;
    description: string;
    material?: string;
    productDescription?: string;
    frameImage?: string;
    details?: string;
}

/**
 * Clothing style categories
 */
export type ClothingStyle = 'Casual' | 'Formal' | 'Sport' | 'Streetwear' | 'Beach' | 'Elegant';

/**
 * AI analysis result from video/image scan
 */
export interface AnalysisResult {
    detectedItems: DetectedItem[];
    frameImage?: string;
    confidence?: number;
}

/**
 * Weather API response
 */
export interface WeatherData {
    temp: number;
    feels_like: number;
    description: string;
    icon: string;
    city: string;
    humidity: number;
    wind_speed: number;
}

/**
 * Image generation request
 */
export interface ImageGenerationRequest {
    prompt: string;
    itemType?: string;
    color?: string;
}

/**
 * Image generation response
 */
export interface ImageGenerationResponse {
    imageUrl: string;
    success: boolean;
}

/**
 * Virtual try-on request
 */
export interface TryOnRequest {
    human_image: string;
    garment_image: string;
    description?: string;
}

/**
 * Virtual try-on response
 */
export interface TryOnResponse {
    image: string;
}

/**
 * Product photo processing response
 */
export interface ProductPhotoResponse {
    success: boolean;
    imageUrl: string;
    bestFrameIndex: number;
    steps: string[];
    style: string;
    preservedOriginal: boolean;
}

/**
 * Smart search result
 */
export interface SearchResult {
    occasion: string;
    style: string;
    items: string[];
    image: string;
    score: number;
}

/**
 * AI chat response
 */
export interface ChatResponse {
    text: string;
}

/**
 * Error response
 */
export interface ErrorResponse {
    error: string;
    details?: string;
}
