export { canonicalizeMacroCategory, getMacroCategory, normalizeCategory } from './categoryMapper';

/**
 * Maps any AI-returned color string to a canonical color ID.
 * Single source of truth — replaces all inline copies.
 */
export const mapColorToId = (colorName: string): string => {
  const name = (colorName || '').toLowerCase();
  if (name.includes('black') || name.includes('charcoal') || name.includes('ebony')) return 'black';
  if (name.includes('white') || name.includes('ivory') || name.includes('off-white') || name.includes('snow')) return 'white';
  if (name.includes('grey') || name.includes('gray') || name.includes('silver') || name.includes('slate')) return 'grey';
  if (name.includes('cream') || name.includes('champagne') || name.includes('vanilla')) return 'cream';
  if (name.includes('beige') || name.includes('tan') || name.includes('khaki') || name.includes('sand') || name.includes('nude')) return 'beige';
  if (name.includes('brown') || name.includes('camel') || name.includes('chocolate') || name.includes('mocha') || name.includes('rust') || name.includes('chestnut')) return 'brown';
  if (name.includes('pink') || name.includes('coral') || name.includes('blush') || name.includes('rose') || name.includes('fuchsia') || name.includes('salmon') || name.includes('magenta')) return 'pink';
  if (name.includes('red') || name.includes('burgundy') || name.includes('wine') || name.includes('scarlet') || name.includes('crimson') || name.includes('maroon')) return 'red';
  if (name.includes('orange') || name.includes('amber') || name.includes('terracotta') || name.includes('burnt orange')) return 'orange';
  if (name.includes('yellow') || name.includes('gold') || name.includes('mustard') || name.includes('lemon')) return 'yellow';
  if (name.includes('olive') || name.includes('forest') || name.includes('sage') || name.includes('mint') || name.includes('lime') || name.includes('green') || name.includes('emerald')) return 'green';
  if (name.includes('teal') || name.includes('turquoise') || name.includes('aqua') || name.includes('cyan') || name.includes('seafoam')) return 'teal';
  if (name.includes('navy') || name.includes('midnight') || name.includes('dark blue')) return 'navy';
  if (name.includes('denim') || name.includes('indigo') || name.includes('cobalt') || name.includes('royal blue') || name.includes('sky blue') || name.includes('blue')) return 'blue';
  if (name.includes('purple') || name.includes('violet') || name.includes('lavender') || name.includes('lilac') || name.includes('plum')) return 'purple';
  if (name.includes('multi') || name.includes('pattern') || name.includes('print') || name.includes('stripe') || name.includes('plaid') || name.includes('floral') || name.includes('checkered')) return 'multicolor';
  
  const direct = ['black','white','grey','beige','cream','brown','red','pink','orange','yellow','green','teal','blue','navy','purple','multicolor'];
  for (const id of direct) { 
    if (name === id) return id; 
  }
  return 'grey'; // Default fallback
};

/**
 * Maps AI category + section to a UI type ID.
 * Replaces mapCategoryToType in MyClosetScreen and the inline mapType in ClothingDetailEditor.
 */
export const mapCategoryToType = (category: string, section: string = ''): string => {
  const cat = (category || '').toLowerCase();
  const sec = (section || '').toLowerCase();
  
  // Specific garment identity wins over stale generic sections.
  if (/trouser|pant|jean|chino|slack|skirt|short|брюк|джинс/.test(cat)) return 'bottoms';
  if (/jacket|coat|blazer|puffer|parka|куртк|пиджак/.test(cat)) return 'outerwear';
  if (/shoe|sneaker|boot|sandal|loafer|кроссов/.test(cat)) return 'shoes';
  if (/shirt|blouse|sweater|hoodie|polo|рубаш|футбол/.test(cat)) return 'tops';
  if (/bag|hat|scarf|belt|watch/.test(cat)) return 'accessories';
  if (/sport|gym|legging/.test(cat)) return 'sportswear';
  const aliases: Record<string, string> = {top:'tops',tops:'tops',upper_body:'tops',bottom:'bottoms',bottoms:'bottoms',lower_body:'bottoms',shoes:'shoes',outerwear:'outerwear',accessory:'accessories',accessories:'accessories',sportswear:'sportswear',homewear:'homewear'};
  if (aliases[cat]) return aliases[cat];
  if (aliases[sec]) return aliases[sec];
  return 'tops'; // Default fallback
};
