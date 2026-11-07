import type { ShopCatalogItem } from '../../types/shop';
import { inferShopRowMacro } from '../../utils/shopCatalogOrder';
import { normalizeIdentityText } from '../../utils/catalogIdentity';
export function fitCategory(value: string): string {
  const key = normalizeIdentityText(value);
  return (
    (
      {
        top: 'top',
        shirt: 'top',
        shirts: 'top',
        polo: 'top',
        overshirt: 'top',
        dresses: 'dress',
        accessories: 'accessory',
        outfits: 'outfit',
        tops: 'top',
        trousers: 'bottom',
        pants: 'bottom',
        'upper body': 'top',
        bottoms: 'bottom',
        'lower body': 'bottom',
        shoe: 'shoes',
        outer: 'outerwear',
      } as Record<string, string>
    )[key] || key
  );
}

export function fitSubcategory(value: string): string | undefined {
  const text = normalizeIdentityText(value);
  for (const [pattern, type] of [
    [/\bovershirts?\b/, 'overshirt'],
    [/\bpolos?\b/, 'polo'],
    [/\b(tee|t shirt|tshirt)\b/, 'tee'],
    [/\b(shirts?|oxford shirt)\b/, 'shirt'],
    [/\bblazers?\b/, 'blazer'],
    [/\bcoats?\b/, 'coat'],
    [/\bjeans?\b/, 'jeans'],
    [/\bchinos?\b/, 'chinos'],
    [/\btrousers?\b|\bpants?\b/, 'trousers'],
    [/\bloafers?\b/, 'loafer'],
    [/\b(sneakers?|trainers?)\b/, 'sneaker'],
    [/\bboots?\b/, 'boot'],
  ] as const)
    if (pattern.test(text)) return type;
  return undefined;
}

export function productFitCategory(item: ShopCatalogItem): string {
  if (item.garmentType === 'dresses') return 'dress';
  if (item.garmentType === 'accessory') return 'accessory';
  if (item.garmentType === 'outfit') return 'outfit';
  return fitCategory(item.category || inferShopRowMacro(item.garmentType, item.name));
}
