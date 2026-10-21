// Gemini selects identifiers; trusted database rows supply product attributes.
export function geminiOutfitSchema(candidateIds: string[], maxOutfits: number) {
  const ids = [...new Set(candidateIds)];
  if (!ids.length || ids.length > 200 || ids.some(id => typeof id !== 'string' || !id || id.length > 200)) {
    throw new Error('invalid_gemini_candidate_ids');
  }
  if (!Number.isInteger(maxOutfits) || maxOutfits < 1 || maxOutfits > 15) throw new Error('invalid_gemini_outfit_count');
  return {
    type: 'object', additionalProperties: false, required: ['outfits'],
    properties: {
      outfits: {
        type: 'array', minItems: 1, maxItems: maxOutfits,
        items: {
          type: 'object', additionalProperties: false,
          required: ['id', 'style', 'occasion', 'description', 'confidence', 'items', 'stylingTips'],
          properties: {
            id: { type: 'string' }, style: { type: 'string' }, occasion: { type: 'string' },
            description: { type: 'string' }, confidence: { type: 'number', minimum: 0, maximum: 1 },
            items: {
              type: 'array', minItems: 2, maxItems: 6,
              items: {
                type: 'object', additionalProperties: false, required: ['id', 'recommendation'],
                properties: { id: { type: 'string', enum: ids }, recommendation: { type: 'string' } },
              },
            },
            stylingTips: { type: 'array', minItems: 2, maxItems: 2, items: { type: 'string' } },
          },
        },
      },
    },
  };
}
