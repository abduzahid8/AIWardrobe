// Designer fashion styles data
export interface DesignerStyle {
    id: string;
    name: string;
    translationKey: string;
    colors: string[];
    icon: string;
    keywords: string[];
}

export const DESIGNER_STYLES: DesignerStyle[] = [
    {
        id: 'classic_refinement',
        name: 'Classic Refinement',
        translationKey: 'styles.classic',
        colors: ['#0A1931', '#757575', '#FFFFFF'],
        icon: '👔',
        keywords: ['timeless', 'gray trousers', 'navy blazer', 'refined', 'elegant']
    },
    {
        id: 'smart_casual',
        name: 'Smart Casual',
        translationKey: 'styles.smart_casual',
        colors: ['#8B4513', '#556B2F', '#D4C4A8'],
        icon: '👟',
        keywords: ['textured', 'knitwear', 'sport jacket', 'relaxed', 'versatile']
    },
    {
        id: 'summer_elegance',
        name: 'Summer Elegance',
        translationKey: 'styles.summer',
        colors: ['#FFFFF0', '#ADD8E6', '#F5F5DC'],
        icon: '☀️',
        keywords: ['linen', 'tropical wool', 'breathable', 'light colors', 'no shorts']
    },
    {
        id: 'lauren',
        name: 'Ralph Lauren (Preppy)',
        translationKey: 'styles.lauren',
        colors: ['#000080', '#FFFFFF', '#8B4513'],
        icon: '🐎',
        keywords: ['preppy', 'classic', 'sporty', 'american', 'casual elegance']
    }
];

export const getStylePromptSuffix = (styleId: string): string => {
    const style = DESIGNER_STYLES.find(s => s.id === styleId);
    if (!style) return '';

    return `Style inspiration: ${style.name}. Keywords: ${style.keywords.join(', ')}.`;
};
