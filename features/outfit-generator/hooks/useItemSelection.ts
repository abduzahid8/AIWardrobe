/**
 * useItemSelection — manages item selection with one-per-category enforcement
 */

import { useState, useCallback } from 'react';

interface SelectableItem {
    id: string;
    type?: string;
    category?: string;
    [key: string]: any;
}

export function getMacroCategory(type: string): string {
    const t = type.toLowerCase();
    if (t.includes('sweater') || t.includes('hoodie') || t.includes('cardigan')) return 'sweater';
    if (t.includes('shirt') || t.includes('tee') || t.includes('top') || t.includes('blouse') || t.includes('polo')) return 'top';
    if (t.includes('pant') || t.includes('jean') || t.includes('trouser') || t.includes('short') || t.includes('skirt')) return 'bottom';
    if (t.includes('shoe') || t.includes('boot') || t.includes('sneaker') || t.includes('sandal') || t.includes('loafer')) return 'shoes';
    if (t.includes('jacket') || t.includes('coat') || t.includes('blazer') || t.includes('vest')) return 'outerwear';
    return 'other';
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
