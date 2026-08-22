-- ============================================================
-- 014_inspo_capsules.sql
-- Add featured outfit capsules to shop_catalog
-- ============================================================
--
-- Adds 3 featured outfit capsule items that are fully editable
-- from the admin panel (AdminManageTab and AdminInspoTab)
-- ============================================================

INSERT INTO public.shop_catalog (id, brand, name, price, currency, category, garment_type, description, image_url, is_active, sort_order, source) VALUES
    (
        'inspo-capsule-01',
        'ZARA',
        'Casual Polo & Drawstring Pants',
        89.9,
        'USD',
        'outfit',
        'outfit',
        'Light grey polo shirt paired with brown drawstring pants and brown boat shoes for a relaxed summer look',
        'https://images.unsplash.com/photo-1617137968427-85924c800a22?w=800&q=80',
        TRUE,
        100,
        'inspo'
    ),
    (
        'inspo-capsule-02',
        'ZARA',
        'Linen Summer Essential',
        129.9,
        'USD',
        'outfit',
        'outfit',
        'White linen shirt with brown linen pants and dark brown flip-flops for effortless beach style',
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80',
        TRUE,
        101,
        'inspo'
    ),
    (
        'inspo-capsule-03',
        'ZARA',
        'Smart Casual Ensemble',
        149.9,
        'USD',
        'outfit',
        'outfit',
        'Sophisticated smart casual outfit perfect for weekend outings',
        'https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?w=800&q=80',
        TRUE,
        102,
        'inspo'
    )
ON CONFLICT (id) DO NOTHING;
