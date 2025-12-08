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
        id: 'dior',
        name: 'Christian Dior',
        translationKey: 'styles.dior',
        colors: ['#B8860B', '#2F4F4F', '#FFFFF0'],
        icon: '👗',
        keywords: ['elegant', 'feminine', 'classic', 'haute couture', 'refined']
    },
    {
        id: 'armani',
        name: 'Giorgio Armani',
        translationKey: 'styles.armani',
        colors: ['#36454F', '#C0C0C0', '#000000'],
        icon: '🎩',
        keywords: ['minimalist', 'sophisticated', 'tailored', 'relaxed', 'luxury']
    },
    {
        id: 'lauren',
        name: 'Ralph Lauren',
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
