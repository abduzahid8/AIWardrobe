export const styleCategories = {
    SEMI_CLASSIC: {
        id: 'semi_classic',
        name: "Semi-Classic",
        philosophy: "The art of non-formal refinement. Aim for balance between structure and ease.",
        rules: [
            "Always match a textured Sport Jacket with contrasting trousers (e.g., Flannel jacket with Cotton trousers).",
            "Prefer Loafers (Suede or Leather) over Oxfords.",
            "If using a tie, it MUST be a knit tie or wool tie—avoid silk formal ties.",
            "Roll up shirt sleeves under the jacket only if at a relaxed social event."
        ],
        palette: ["Navy", "Forest Green", "Gray", "Tobacco Brown"],
        proportions: "Natural shoulder (unstructured) for the jacket, standard rise for trousers.",
        avoid: ["Complete Suits", "Sneakers", "Graphic anything", "Silk Bow Ties"]
    },
    MINIMALIST: {
        id: 'minimalist',
        name: "Minimalist Class",
        philosophy: "Maximum impact with minimum items. The 10-Item Capsule logic.",
        rules: [
            "Zero visible logos. The quality of the fabric (Cashmere, Sea Island Cotton) is the statement.",
            "Stick strictly to the core 10: Navy Blazer, Gray Trousers, White Shirt, etc.",
            "Items must be monochrome or tonal (shades of the same color).",
            "Ensure perfectly tailored fit—minimalism fails if the item is baggy or ill-fitting."
        ],
        palette: ["Charcoal", "Navy", "White", "Black (only for evening)"],
        proportions: "Clean lines. No pleats on trousers. Slim or tailored straight fits.",
        avoid: ["Patterns (stripes/checks)", "Excessive accessories", "Cheap synthetics"]
    },
    CASUAL: {
        id: 'casual',
        name: "Refined Casual",
        philosophy: "Effortless style for the city. Not for the gym.",
        rules: [
            "Knitted/Woven Polos are the standard. Basic cotton T-shirts are only for sports.",
            "Never wear shorts in the city. Use Linen or Tropical Wool trousers instead.",
            "Minimalist White Leather Sneakers are the only allowed athletic shoe.",
            "Outerwear should be a bomber, harrington, or a high-quality cardigan—never a hoodie."
        ],
        palette: ["Beige", "Olive", "Light Blue", "Off-white"],
        proportions: "Relaxed but fitted. High-waisted chinos create the best silhouette.",
        avoid: ["Shorts", "Hoodies", "Joggers", "Ankle socks (use footies)"]
    },
    BUSINESS_CASUAL: {
        id: 'business_casual',
        name: "Modern Professional",
        philosophy: "Sharp, authoritative style for the contemporary high-end workplace.",
        rules: [
            "The jacket is mandatory. If not a blazer, use a structured cardigan.",
            "High-waisted trousers with side adjusters (no belt) are the elite standard.",
            "Shirts must be crisp (Poplin or Oxford) and always tucked in.",
            "Match watch face size to wrist—large 'cringy' watches ruin the professional look."
        ],
        palette: ["Midnight Blue", "Medium Gray", "Burgundy", "Camel"],
        proportions: "Structured shoulders. Tapered trousers with a slight break.",
        avoid: ["Jeans", "T-shirts under blazers", "Backpacks (use leather briefcase)", "Rubber soles"]
    }
};

export const getCategoryByStyleId = (styleId) => {
    const mapping = {
        'old_money': styleCategories.MINIMALIST, // Old Money shares DNA with Minimalist refinement
        'semi_classic': styleCategories.SEMI_CLASSIC,
        'minimalist': styleCategories.MINIMALIST,
        'casual': styleCategories.CASUAL,
        'business_casual': styleCategories.BUSINESS_CASUAL,
        'summer_elegance': styleCategories.CASUAL, // Summer is a subset of Refined Casual logic
    };
    return mapping[styleId] || styleCategories.CASUAL;
};
