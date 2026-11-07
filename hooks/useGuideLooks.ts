import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';

export interface GuideLookProduct {
    id: string;
    look_id: string;
    name: string;
    category: string;
    image_url: string;
    price: number | null;
    currency: string | null;
    product_url: string;
    sort_order: number;
}

export interface GuideLook {
    id: string;
    brand: string;
    collection_name: string;
    title: string;
    image_url: string;
    source_url: string;
    sort_order: number;
    products: GuideLookProduct[];
}

export function useGuideLooks() {
    const [items, setItems] = useState<GuideLook[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        const { data: looks, error: looksError } = await supabase
            .from('inspo_looks')
            .select('id, brand, collection_name, title, image_url, source_url, sort_order')
            .eq('is_active', true)
            .order('sort_order', { ascending: true });
        if (looksError) {
            setError(looksError.message);
            setLoading(false);
            return;
        }

        const lookRows = looks ?? [];
        setError(null);
        if (!lookRows.length) {
            setItems([]);
            setLoading(false);
            return;
        }

        const { data: products, error: productsError } = await supabase
            .from('inspo_look_products')
            .select('id, look_id, name, category, image_url, price, currency, product_url, sort_order')
            .in('look_id', lookRows.map((look: any) => look.id))
            .eq('is_active', true)
            .order('sort_order', { ascending: true });
        if (productsError) {
            setError(productsError.message);
            setLoading(false);
            return;
        }

        const productsByLook = new Map<string, GuideLookProduct[]>();
        (products ?? []).forEach((product: GuideLookProduct) => {
            const current = productsByLook.get(product.look_id) ?? [];
            current.push(product);
            productsByLook.set(product.look_id, current);
        });
        setItems(lookRows.map((look: any) => ({ ...look, products: productsByLook.get(look.id) ?? [] })));
        setLoading(false);
    }, []);

    useEffect(() => { void refresh(); }, [refresh]);
    return { items, loading, error, refresh };
}
