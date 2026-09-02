/**
 * Style Quiz Step Data — All quiz option constants.
 *
 * Extracted from StyleQuizScreen to keep the screen lean
 * and make quiz options easy to update independently.
 */

export const STYLE_PERSONALITIES = [
    { id: 'classic', name: 'Classic', emoji: '👔', description: 'Timeless, elegant, refined', tKey: 'styleQuiz.personality.classic', tDescKey: 'styleQuiz.personality.classicDesc' },
    { id: 'semi_classic', name: 'Semi-Classic', emoji: '🧥', description: 'Tailored comfort, everyday refinement', tKey: 'styleQuiz.personality.semiClassic', tDescKey: 'styleQuiz.personality.semiClassicDesc' },
    { id: 'minimalist', name: 'Minimalist', emoji: '🤍', description: 'Clean lines, simple', tKey: 'styleQuiz.personality.minimalist', tDescKey: 'styleQuiz.personality.minimalistDesc' },
    { id: 'casual', name: 'Casual', emoji: '👕', description: 'Relaxed, effortless, clean daily wear', tKey: 'styleQuiz.personality.casual', tDescKey: 'styleQuiz.personality.casualDesc' },
    { id: 'old_money', name: 'Old Money', emoji: '⚜️', description: 'Quiet luxury, heritage sophistication', tKey: 'styleQuiz.personality.oldMoney', tDescKey: 'styleQuiz.personality.oldMoneyDesc' },
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
    { id: 'work', name: 'Work/Office', icon: 'briefcase-outline', tKey: 'styleQuiz.occasions.work' },
    { id: 'casual', name: 'Casual/Weekend', icon: 'cafe-outline', tKey: 'styleQuiz.occasions.casual' },
    { id: 'date', name: 'Date Night', icon: 'heart-outline', tKey: 'styleQuiz.occasions.date' },
    { id: 'formal', name: 'Formal Events', icon: 'diamond-outline', tKey: 'styleQuiz.occasions.formal' },
] as const;

export const FIT_OPTIONS = [
    { id: 'loose', name: 'Relaxed Fit', description: 'Comfortable, roomy', icon: '👕', tKey: 'styleQuiz.fit.relaxed', tDescKey: 'styleQuiz.fit.relaxedDesc' },
    { id: 'fitted', name: 'Fitted', description: 'Tailored, structured', icon: '👔', tKey: 'styleQuiz.fit.fitted', tDescKey: 'styleQuiz.fit.fittedDesc' },
    { id: 'balanced', name: 'Balanced', description: 'Mix of both', icon: '⚖️', tKey: 'styleQuiz.fit.balanced', tDescKey: 'styleQuiz.fit.balancedDesc' },
] as const;

export const STYLE_GOALS = [
    { id: 'organize_closet', name: 'Organize My Closet', icon: 'grid-outline', tKey: 'styleQuiz.goals.organizeCloset' },
    { id: 'get_styled', name: 'Get AI Outfit Ideas', icon: 'sparkles-outline', tKey: 'styleQuiz.goals.getStyled' },
    { id: 'shop_smarter', name: 'Shop Smarter', icon: 'cart-outline', tKey: 'styleQuiz.goals.shopSmarter' },
    { id: 'build_capsule', name: 'Build a Capsule Wardrobe', icon: 'cube-outline', tKey: 'styleQuiz.goals.buildCapsule' },
    { id: 'explore_trends', name: 'Explore New Trends', icon: 'trending-up-outline', tKey: 'styleQuiz.goals.exploreTrends' },
    { id: 'sustainability', name: 'Be More Sustainable', icon: 'leaf-outline', tKey: 'styleQuiz.goals.sustainability' },
] as const;
