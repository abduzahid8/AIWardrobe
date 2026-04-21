import { validateClothingRows, ClothingItemRowSchema } from '../../src/utils/validators';

describe('validators', () => {
    const validRow = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        user_id: '550e8400-e29b-41d4-a716-446655440001',
        image_url: 'https://example.com/image.jpg',
        category: 'top',
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
    };

    it('validates a correct clothing row', () => {
        const result = ClothingItemRowSchema.safeParse(validRow);
        expect(result.success).toBe(true);
    });

    it('rejects row with invalid UUID', () => {
        const result = ClothingItemRowSchema.safeParse({ ...validRow, id: 'not-a-uuid' });
        expect(result.success).toBe(false);
    });

    it('rejects row with invalid category', () => {
        const result = ClothingItemRowSchema.safeParse({ ...validRow, category: 'pants' });
        expect(result.success).toBe(false);
    });

    it('rejects row with missing image_url', () => {
        const result = ClothingItemRowSchema.safeParse({ ...validRow, image_url: '' });
        expect(result.success).toBe(false);
    });

    describe('validateClothingRows', () => {
        it('filters out invalid rows', () => {
            const rows = [
                validRow,
                { ...validRow, id: 'bad-id' },
                { ...validRow, id: '660e8400-e29b-41d4-a716-446655440002' },
            ];
            const valid = validateClothingRows(rows);
            expect(valid.length).toBe(2);
        });

        it('returns empty array for all-invalid input', () => {
            expect(validateClothingRows([{ garbage: true }])).toEqual([]);
        });
    });
});
