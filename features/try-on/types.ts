/**
 * Try-On feature — shared types
 */

export interface WardrobeItem {
    _id: string;
    id?: string;
    type?: string;
    itemType?: string;
    category?: string;
    color?: string;
    imageUrl?: string;
    image?: string;
}

export type TryOnMode = 'try your self' | 'model';
export type TryOnStep = 1 | 2 | 3;
export type PhotoTab = 'upload' | 'wardrobe';
