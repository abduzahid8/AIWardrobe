-- ============================================================
-- 025_repair_featured_capsule_images.sql
-- Repair featured_capsules rows pointing at deleted storage objects.
-- Run in: Supabase Dashboard → SQL Editor
-- ============================================================
--
-- Background: the three capsule rows below referenced objects under
-- `shop-catalog/admin/` that no longer exist (the whole `admin/` prefix is
-- empty — every fetch returns 404 NoSuchKey), so the Inspo Guide + Shop tabs
-- rendered blank placeholder cards on fresh installs. The app ships a
-- client-side shim for these exact URLs (see useFeaturedCapsules.ts), but
-- repairing the rows is the permanent fix — after this runs, the shim
-- becomes a no-op.
--
-- Each replacement image was verified to return `200` with an image
-- content-type before this migration was written.
-- The `image_url LIKE` guard means this only touches still-broken rows and
-- never overwrites images an admin re-uploaded after the deletion.

UPDATE public.featured_capsules
SET image_url = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop-catalog/items/shop-inspo-1.png'
WHERE id = '66da23a0-ab32-43e7-8c61-7208a0d6caf2'   -- Summer Outfits
  AND image_url LIKE '%admin/capsule-1781186443576.jpg%';

UPDATE public.featured_capsules
SET image_url = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop-catalog/items/shop-inspo-3.png'
WHERE id = '71038c57-5e1c-48b7-acd8-5d6f1f9e689b'   -- Summer 2026
  AND image_url LIKE '%admin/capsule-1781186470011.jpg%';

UPDATE public.featured_capsules
SET image_url = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop-catalog/items/shop-inspo-5.png'
WHERE id = '07fa22e6-756e-4b16-af85-2dc5e141b517'   -- Massimo Dutti
  AND image_url LIKE '%admin/capsule-1781186613082.png%';
