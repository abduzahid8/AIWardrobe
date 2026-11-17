/** Same deterministic golden inputs in Node/Expo-compatible TS and Deno. */
import { stylingContext } from '../../src/domain/stylingContext.ts';
import { outfitCoherence } from '../../src/domain/outfitCoherence.ts';
import { outfitVisualSignature } from '../../src/domain/outfitVisualSignature.ts';
import { fourSlotOutfitReason } from '../../src/domain/outfitStructure.ts';
const inputs = [
  {
    style: 'semi_classic',
    occasion: 'Formal',
    names: [
      'White poplin dress shirt',
      'Wool blazer',
      'Tailored wool trousers',
      'Black leather Oxford shoes',
    ],
  },
  {
    style: 'minimalist',
    occasion: 'Date night',
    names: [
      'White cotton shirt',
      'Fine merino cardigan',
      'Grey tailored trousers',
      'Waterproof hiking boots with lug sole',
    ],
  },
  {
    style: 'casual',
    occasion: 'Weekend',
    names: [
      'Cotton T-shirt',
      'Lightweight linen overshirt',
      'Straight jeans',
      'Clean leather sneakers',
    ],
  },
];
console.log(
  JSON.stringify(
    inputs.map((input) => {
      const items = input.names.map((name, i) => ({
        id: `piece-${i}`,
        name,
        macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][i],
      }));
      return {
        context: stylingContext(input),
        structure: fourSlotOutfitReason(items),
        coherence: outfitCoherence(items, stylingContext(input)),
        signature: outfitVisualSignature(items),
      };
    })
  )
);
