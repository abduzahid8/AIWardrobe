/** Explainable soft preferences; dress-code hard gates live in garmentContext. */
export interface StylingProfile {
  preferred: readonly string[];
  textures: readonly string[];
  restrained: boolean;
}
export const STYLING_PROFILES: Readonly<Record<string, StylingProfile>> = {
  minimalist: {
    preferred: [
      'shirt',
      'fine-knit',
      'cardigan',
      'overshirt',
      'tailored-trouser',
      'chino',
      'minimal-sneaker',
      'chelsea',
      'dress-boot',
    ],
    textures: ['smooth', 'fine-knit'],
    restrained: true,
  },
  old_money: {
    preferred: ['polo', 'dress-shirt', 'cardigan', 'blazer', 'chino', 'tailored-trouser', 'loafer'],
    textures: ['suede', 'knit', 'fine-knit', 'wool'],
    restrained: true,
  },
  semi_classic: {
    preferred: [
      'dress-shirt',
      'shirt',
      'blazer',
      'tailored-trouser',
      'derby',
      'oxford',
      'monk',
      'dress-boot',
    ],
    textures: ['smooth', 'wool'],
    restrained: false,
  },
  casual: {
    preferred: [
      't-shirt',
      'polo',
      'shirt',
      'overshirt',
      'jacket',
      'bomber',
      'jeans',
      'chino',
      'sneaker',
      'minimal-sneaker',
    ],
    textures: ['denim', 'cotton'],
    restrained: false,
  },
};
