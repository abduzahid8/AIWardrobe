/**
 * Style Quiz Step Data — All quiz option constants.
 *
 * Extracted from StyleQuizScreen to keep the screen lean
 * and make quiz options easy to update independently.
 */

export const STYLE_PERSONALITIES = [
    { id: 'classic', name: 'Classic', emoji: '👔', description: 'Timeless, elegant, refined' },
    { id: 'trendy', name: 'Trendy', emoji: '✨', description: 'Fashion-forward, current' },
    { id: 'minimalist', name: 'Minimalist', emoji: '⬜', description: 'Clean lines, simple' },
    { id: 'bohemian', name: 'Bohemian', emoji: '🌸', description: 'Free-spirited, artistic' },
    { id: 'edgy', name: 'Edgy', emoji: '🖤', description: 'Bold, unconventional' },
    { id: 'romantic', name: 'Romantic', emoji: '💕', description: 'Soft, feminine, flowy' },
    { id: 'sporty', name: 'Sporty', emoji: '⚡', description: 'Active, comfortable' },
] as const;

export const COLOR_OPTIONS = [
    { id: '#0A1931', name: '#0A1931', color: '#0A1931' },
    { id: 'white', name: 'White', color: '#FFFFFF' },
    { id: 'navy', name: 'Navy', color: '#1a237e' },
    { id: 'beige', name: 'Beige', color: '#d4c4a8' },
    { id: 'gray', name: 'Gray', color: '#757575' },
    { id: 'burgundy', name: 'Burgundy', color: '#800020' },
    { id: 'olive', name: 'Olive', color: '#556b2f' },
    { id: 'brown', name: 'Brown', color: '#8b4513' },
    { id: 'pink', name: 'Pink', color: '#e91e63' },
    { id: 'blue', name: 'Blue', color: '#2196f3' },
    { id: 'red', name: 'Red', color: '#f44336' },
    { id: 'green', name: 'Green', color: '#4caf50' },
] as const;

export const OCCASIONS = [
    { id: 'work', name: 'Work/Office', icon: 'briefcase-outline' },
    { id: 'casual', name: 'Casual/Weekend', icon: 'cafe-outline' },
    { id: 'date', name: 'Date Night', icon: 'heart-outline' },
    { id: 'fitness', name: 'Fitness/Gym', icon: 'fitness-outline' },
    { id: 'formal', name: 'Formal Events', icon: 'diamond-outline' },
    { id: 'travel', name: 'Travel', icon: 'airplane-outline' },
] as const;

export const FIT_OPTIONS = [
    { id: 'loose', name: 'Relaxed Fit', description: 'Comfortable, roomy', icon: '👕' },
    { id: 'fitted', name: 'Fitted', description: 'Tailored, structured', icon: '👔' },
    { id: 'balanced', name: 'Balanced', description: 'Mix of both', icon: '⚖️' },
] as const;

export const STYLE_GOALS = [
    { id: 'organize_closet', name: 'Organize My Closet', icon: 'grid-outline' },
    { id: 'get_styled', name: 'Get AI Outfit Ideas', icon: 'sparkles-outline' },
    { id: 'shop_smarter', name: 'Shop Smarter', icon: 'cart-outline' },
    { id: 'build_capsule', name: 'Build a Capsule Wardrobe', icon: 'cube-outline' },
    { id: 'explore_trends', name: 'Explore New Trends', icon: 'trending-up-outline' },
    { id: 'sustainability', name: 'Be More Sustainable', icon: 'leaf-outline' },
] as const;
