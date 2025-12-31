/**
 * Occasion Templates
 * Pre-built outfit prompts for common occasions
 * Makes AI outfit generation faster and more accurate
 */

export interface OccasionTemplate {
    id: string;
    emoji: string;
    title: string;
    subtitle: string;
    prompt: string;
    suggestedItems?: string[];
    weather?: 'hot' | 'cold' | 'moderate' | 'any';
    formality?: 'casual' | 'business' | 'formal' | 'athletic';
    gradient: [string, string];
}

export const occasionTemplates: OccasionTemplate[] = [
    // Casual & Social
    {
        id: 'coffee_date',
        emoji: '☕',
        title: 'Coffee Date',
        subtitle: 'Casual & approachable',
        prompt: 'Create a casual, comfortable outfit for a coffee date. Something approachable but stylish, perfect for a relaxed conversation.',
        suggestedItems: ['jeans', 'casual_top', 'sneakers', 'jacket'],
        formality: 'casual',
        gradient: ['#C67C4E', '#EDD6C8']
    },
    {
        id: 'date_night',
        emoji: '💃',
        title: 'Date Night',
        subtitle: 'Romantic & elegant',
        prompt: 'Design a romantic date night outfit for an upscale restaurant or special evening. Something that makes me feel confident and attractive.',
        suggestedItems: ['dress', 'heels', 'blazer', 'dress_pants'],
        formality: 'formal',
        gradient: ['#E74C3C', '#F8B500']
    },
    {
        id: 'brunch',
        emoji: '🥂',
        title: 'Brunch',
        subtitle: 'Stylish & comfortable',
        prompt: 'Put together a chic brunch outfit that\'s Instagram-worthy but comfortable for a long meal with friends.',
        suggestedItems: ['sundress', 'blouse', 'sandals', 'accessories'],
        formality: 'casual',
        gradient: ['#F39C12', '#F8E9A1']
    },

    // Work & Business
    {
        id: 'job_interview',
        emoji: '💼',
        title: 'Job Interview',
        subtitle: 'Professional & confident',
        prompt: 'Create a professional interview outfit that projects confidence and competence. Business formal attire that makes a strong first impression.',
        suggestedItems: ['blazer', 'dress_pants', 'button_shirt', 'dress_shoes'],
        formality: 'formal',
        gradient: ['#2C3E50', '#4CA1AF']
    },
    {
        id: 'office_day',
        emoji: '📊',
        title: 'Office Day',
        subtitle: 'Business casual',
        prompt: 'Design a business casual office outfit that\'s professional but comfortable for a full day of work.',
        suggestedItems: ['slacks', 'blouse', 'loafers', 'cardigan'],
        formality: 'business',
        gradient: ['#667EEA', '#764BA2']
    },
    {
        id: 'presentation',
        emoji: '🎤',
        title: 'Presentation',
        subtitle: 'Polished & authoritative',
        prompt: 'Create an outfit for giving an important presentation. Something that commands attention and projects authority.',
        suggestedItems: ['suit', 'dress_shirt', 'formal_shoes'],
        formality: 'formal',
        gradient: ['#134E5E', '#71B280']
    },

    // Active & Outdoor
    {
        id: 'gym',
        emoji: '💪',
        title: 'Gym Workout',
        subtitle: 'Performance & comfort',
        prompt: 'Put together a comfortable, functional gym outfit perfect for my workout routine.',
        suggestedItems: ['leggings', 'sports_bra', 'sneakers', 'tank_top'],
        formality: 'athletic',
        gradient: ['#06BEB6', '#48B1BF']
    },
    {
        id: 'hiking',
        emoji: '🥾',
        title: 'Hiking Trip',
        subtitle: 'Outdoor adventure',
        prompt: 'Design a practical hiking outfit with layers, comfortable shoes, and weather-appropriate clothing.',
        suggestedItems: ['hiking_boots', 'athletic_pants', 'jacket', 'hat'],
        formality: 'athletic',
        gradient: ['#56AB2F', '#A8E063']
    },
    {
        id: 'beach',
        emoji: '🏖️',
        title: 'Beach Day',
        subtitle: 'Sun & fun',
        prompt: 'Create a beach vacation outfit with swimwear, cover-up, and sun protection for a day by the ocean.',
        suggestedItems: ['swimwear', 'cover_up', 'sandals', 'sunglasses', 'hat'],
        weather: 'hot',
        formality: 'casual',
        gradient: ['#4FACFE', '#00F2FE']
    },

    // Events & Celebrations
    {
        id: 'wedding_guest',
        emoji: '💒',
        title: 'Wedding Guest',
        subtitle: 'Elegant celebration',
        prompt: 'Design an elegant outfit appropriate for attending a wedding as a guest. Formal but not outshining the couple.',
        suggestedItems: ['cocktail_dress', 'heels', 'clutch'],
        formality: 'formal',
        gradient: ['#F857A6', '#FF5858']
    },
    {
        id: 'birthday_party',
        emoji: '🎂',
        title: 'Birthday Party',
        subtitle: 'Fun & festive',
        prompt: 'Put together a fun, festive outfit for a birthday celebration. Something photo-worthy and comfortable for partying.',
        suggestedItems: ['party_dress', 'heels', 'statement_accessories'],
        formality: 'casual',
        gradient: ['#FA709A', '#FEE140']
    },
    {
        id: 'concert',
        emoji: '🎶',
        title: 'Concert',
        subtitle: 'Edgy & comfortable',
        prompt: 'Create a stylish concert outfit that\'s comfortable for standing/dancing and matches the music vibe.',
        suggestedItems: ['jeans', 't_shirt', 'boots', 'jacket'],
        formality: 'casual',
        gradient: ['#8E2DE2', '#4A00E0']
    },

    // Travel
    {
        id: 'airport',
        emoji: '✈️',
        title: 'Airport Outfit',
        subtitle: 'Comfortable travel',
        prompt: 'Design a comfortable airport outfit with layers for changing AC temperatures, easy to move in, and stylish.',
        suggestedItems: ['leggings', 'hoodie', 'sneakers', 'jacket'],
        formality: 'casual',
        gradient: ['#36D1DC', '#5B86E5']
    },
    {
        id: 'city_exploring',
        emoji: '🗺️',
        title: 'City Exploring',
        subtitle: 'Walking & sightseeing',
        prompt: 'Create a comfortable outfit for a day of city exploration with lots of walking and sightseeing.',
        suggestedItems: ['comfortable_pants', 'walking_shoes', 'backpack', 'sunglasses'],
        formality: 'casual',
        gradient: ['#FFD89B', '#19547B']
    },

    // Seasons
    {
        id: 'winter_cozy',
        emoji: '❄️',
        title: 'Winter Cozy',
        subtitle: 'Warm & stylish',
        prompt: 'Design a cozy winter outfit with warm layers, perfect for cold weather while still looking fashionable.',
        suggestedItems: ['coat', 'sweater', 'boots', 'scarf'],
        weather: 'cold',
        formality: 'casual',
        gradient: ['#2E3192', '#1BFFFF']
    },
    {
        id: 'summer_casual',
        emoji: '☀️',
        title: 'Summer Casual',
        subtitle: 'Light & breezy',
        prompt: 'Create a light, breathable summer outfit perfect for hot weather. Comfortable and cool.',
        suggestedItems: ['shorts', 'tank_top', 'sandals', 'sunglasses'],
        weather: 'hot',
        formality: 'casual',
        gradient: ['#F2994A', '#F2C94C']
    }
];

export const getTemplateById = (id: string): OccasionTemplate | undefined => {
    return occasionTemplates.find(t => t.id === id);
};

export const getTemplatesByFormality = (formality: string): OccasionTemplate[] => {
    return occasionTemplates.filter(t => t.formality === formality);
};

export const getTemplatesByWeather = (weather: string): OccasionTemplate[] => {
    return occasionTemplates.filter(t => t.weather === weather || t.weather === 'any');
};
