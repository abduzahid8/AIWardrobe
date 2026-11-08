const fs = require('node:fs');
const path = require('node:path');
const { prepareGarment } = require('../../src/domain/garmentDescriptor');

const migrationPath = path.join(
  process.cwd(),
  'backend/supabase/migrations/20261108140000_stylist5_compact_retrieval.sql'
);
const edgePath = path.join(process.cwd(), 'backend/supabase/functions/generate-outfits/index.ts');

test('compact RPC does not emit a partial full-descriptor cache', () => {
  const sql = fs.readFileSync(migrationPath, 'utf8');
  const compact = sql.split(
    'drop function if exists public.catalog_stylist_candidate_hydration'
  )[0];
  expect(compact).not.toContain("'fashion_intelligence'");
  expect(compact).toContain("'price_peers'");
  expect(compact).toContain('concept_reserved');
  expect(compact).toContain('b.cost is not null and b.cost<=cap');
});

test('hydration contract preserves canonical identity and full peer shape', () => {
  const sql = fs.readFileSync(migrationPath, 'utf8');
  const hydration = sql.slice(
    sql.indexOf('create or replace function public.catalog_stylist_candidate_hydration')
  );
  for (const field of [
    'median',
    'p25',
    'p75',
    'count',
    'normalizedPrice',
    'normalizedMedian',
    'normalizedCount',
  ]) {
    expect(hydration).toContain(`'${field}'`);
  }
  const edge = fs.readFileSync(edgePath, 'utf8');
  expect(edge).toMatch(/id:\s*`shop_\$\{row\.id\}`/)
  expect(edge).toContain('Array.isArray(record.descriptor?.evidence)')
  expect(edge).toMatch(/prepareHydratedStylist5Garment\(item,\s*detail\)/)
  expect(edge).toContain('pricePeers:item.pricePeers||detail.price_peers')
  expect(edge).not.toContain('...item,...detail');
});

test('compact catalog rows remain safe for descriptor inference', () => {
  const compactRow = {
    id: 'cardigan-1',
    name: 'Merino cardigan',
    description: 'Fine knit cardigan, product only, plain background',
    garment_type: 'outerwear',
    category: 'outerwear',
    fashion_slot: 'outerwear',
    subtype: 'cardigan',
    formality: 'smart',
    color: 'navy',
    volume: 'regular',
    image_url: 'https://cdn.example/cardigan.jpg',
  };
  expect(compactRow.fashion_intelligence).toBeUndefined();
  expect(
    prepareGarment({
      id: 'shop_cardigan-1',
      name: compactRow.name,
      description: compactRow.description,
      type: compactRow.garment_type,
      macroCategory: compactRow.fashion_slot,
      imageUrl: compactRow.image_url,
    })
  ).toBeDefined();
});
