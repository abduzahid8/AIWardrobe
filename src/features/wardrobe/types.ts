/**
 * Wardrobe Video Analysis — shared types
 * Extracted from WardrobeVideoScreen to enable reuse and testability.
 */

export interface DetectedItem {
    itemType: string;
    specificType?: string;
    classificationPath?: string;
    color: string;
    colorHex?: string;
    style: string;
    description: string;
    material?: string;
    materialDetails?: {
        type: string;
        category: string;
        texture: string;
        finish: string;
        weight: string;
        isStretch: boolean;
    };
    pattern?: string;
    patternDetails?: {
        type: string;
        category: string;
        isStriped: boolean;
        isCheckered: boolean;
        hasPrint: boolean;
        colors: string[];
    };
    neckline?: string;
    sleeveType?: string;
    fit?: string;
    closure?: string;
    details?: string;
    productDescription?: string;
    frameImage?: string;
    position?: string;
    confidence?: number;
    confidenceLevel?: string;
    agreementScore?: number;
    detectionSources?: string[];
    styleTags?: string[];
    features?: Record<string, string | number | boolean>;
    bbox?: number[];
    attributes?: Record<string, string | number | boolean>;
    outfitId?: number;
    framesDetected?: number;
    trackId?: number;
    cutoutImage?: string;
    detectionBox?: number[];
    startFrame?: number;
    frameIndex?: number;
}

export interface APIItemResponse {
    category?: string;
    specificType?: string;
    primaryColor?: string;
    color?: string;
    colorHex?: string;
    material?: string;
    pattern?: string;
    confidence?: number;
    bbox?: number[];
    cutoutImage?: string;
    bestFrame?: string;
    attributes?: Record<string, string | number | boolean>;
    type?: string;
    fit?: string;
    trackId?: number;
    outfitId?: number;
    outfit_id?: number;
    neckline?: string;
    sleeveType?: string;
    styleTags?: string[];
    caption?: string;
    framesDetected?: number;
    frameIndices?: number[];
    identity?: { type?: string; subType?: string; brandGuess?: string };
    construction?: { closure?: string; neckline?: string; sleeves?: string; pockets?: string; details?: string };
    quality?: { condition?: string; level?: string; priceRange?: string };
    position?: string;
    startFrame?: number;
}

export interface OutfitResponse {
    outfitId?: number;
    startFrame?: number;
    items?: APIItemResponse[];
}

export interface AnalysisResult {
    detectedItems: DetectedItem[];
    frameImage?: string;
}
