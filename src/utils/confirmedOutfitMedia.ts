import type { SavedOutfitInput } from '../services/savedOutfitService';
import type { Outfit } from '../types/domain';

/** Match the exact confirmed snapshot, never associate images by position alone. */
export function confirmedImageReplacements(before: SavedOutfitInput['items'], after: SavedOutfitInput['items']) {
  if (before.length !== after.length || before.some((item,index) => item.id !== after[index].id)) {
    throw new Error('Confirmed outfit items do not match');
  }
  const replacements = new Map<string,string>();
  before.forEach((item,index) => {
    for (const field of ['image','imageUrl'] as const) {
      const source = item[field], destination = after[index][field];
      if (source && destination && source !== destination) replacements.set(source,destination);
    }
  });
  return replacements;
}

export function replaceGeneratedOutfitImages<T extends {mainImage:string;items:Array<{image:string;imageUrl?:string}>}>(outfit:T, replacements:ReadonlyMap<string,string>):T {
  if (!replacements.size) return outfit;
  return {...outfit,mainImage:replacements.get(outfit.mainImage) ?? outfit.mainImage,
    items:outfit.items.map(item=>({...item,image:replacements.get(item.image) ?? item.image,
      ...(item.imageUrl ? {imageUrl:replacements.get(item.imageUrl) ?? item.imageUrl}:{}),
    })),
  };
}

export function replaceSavedOutfitImages<T extends Pick<Outfit,'previewImageUrl'|'snapshotItems'>>(outfit:T,replacements:ReadonlyMap<string,string>):T {
  if (!replacements.size) return outfit;
  return {...outfit,
    ...(outfit.previewImageUrl ? {previewImageUrl:replacements.get(outfit.previewImageUrl) ?? outfit.previewImageUrl}:{}),
    ...(outfit.snapshotItems ? {snapshotItems:outfit.snapshotItems.map(item=>replacements.has(item.image)
      ? {...item,image:replacements.get(item.image)!}:item)}:{}),
  };
}
