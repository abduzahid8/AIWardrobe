/**
 * Try-On feature — shared types
 */

export interface WardrobeItem {
    id: string;
    type?: string;
    category?: string;
    color?: string;
    imageUrl?: string;
}

export type TryOnMode = 'try your self' | 'model';
export type TryOnStep = 1 | 2 | 3;
export type PhotoTab = 'upload' | 'wardrobe';
