-- ============================================================
-- AIWardrobe — Migration 003: Align clothing_items categories
-- Converts PascalCase categories to lowercase domain types.
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- Step 1: Migrate existing data to lowercase values
UPDATE public.clothing_items SET category = 'top'       WHERE category = 'Tops';
UPDATE public.clothing_items SET category = 'bottom'    WHERE category = 'Bottoms';
UPDATE public.clothing_items SET category = 'dress'     WHERE category = 'Dresses';
UPDATE public.clothing_items SET category = 'outerwear' WHERE category = 'Outerwear';
UPDATE public.clothing_items SET category = 'shoes'     WHERE category = 'Shoes';
UPDATE public.clothing_items SET category = 'accessory' WHERE category = 'Accessories';
UPDATE public.clothing_items SET category = 'other'     WHERE category = 'Other';

-- Step 2: Drop old constraint and add new one with lowercase values
ALTER TABLE public.clothing_items DROP CONSTRAINT IF EXISTS clothing_items_category_check;
ALTER TABLE public.clothing_items
    ADD CONSTRAINT clothing_items_category_check
    CHECK (category IN ('top', 'bottom', 'dress', 'shoes', 'outerwear', 'accessory', 'other'));

-- Step 3: Update default
ALTER TABLE public.clothing_items ALTER COLUMN category SET DEFAULT 'other';
