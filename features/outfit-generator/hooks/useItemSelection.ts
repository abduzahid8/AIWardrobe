/**
 * useItemSelection — manages item selection with one-per-category enforcement
 */

import { useState, useCallback } from 'react';
import { getMacroCategory } from '@/utils/categoryMapper';

export { getMacroCategory } from '@/utils/categoryMapper';

interface SelectableItem {
    id: string;
    type?: string;
    category?: string;
    [key: string]: any;
}

export function useItemSelection() {
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

    const toggleItemSelection = useCallback((id: string, allItems: SelectableItem[]) => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
                return next;
            }

            // Enforce one-per-category
            const item = allItems.find((i) => i.id === id);
            if (!item) return prev;
            const category = getMacroCategory(item.type || item.category || 'other');

            // Remove existing item in same category
            for (const existingId of next) {
                const existing = allItems.find((i) => i.id === existingId);
                if (existing && getMacroCategory(existing.type || existing.category || 'other') === category) {
                    next.delete(existingId);
                }
            }

            next.add(id);
            return next;
        });
    }, []);

    const clearSelection = useCallback(() => setSelectedIds(new Set()), []);
    const isSelected = useCallback((id: string) => selectedIds.has(id), [selectedIds]);

    return {
        selectedIds,
        toggleItemSelection,
        clearSelection,
        isSelected,
        selectedCount: selectedIds.size,
    };
}
