import test from 'node:test';
import assert from 'node:assert/strict';
import { frColorFromCanonicalUrl } from './merchant_color_fallback.mjs';
import { inferCatalogColor, projectCatalogItem } from './catalog_adapter.mjs';

const make = (name, suffix, attrs={}) => ({
  id: 'fr-test', name, description: 'offline review', brand: 'Tommy Hilfiger',
  category: 'shoes', garment_type: 'shoes', is_active: true,
  source_url: 'https://frgroup.uz/ru/store/lofery-model-fm0fm05891-' + suffix,
  image_url: 'https://media.frgroup.uz/images/photo.webp',
  currency: 'UZS', price: 2050300, ...attrs,
});

test('seller product path grounds missing shade without guessing from brand or arbitrary URL', () => {
  for (const [suffix,color] of [
    ['chernyj','black'],['korichnevyj','brown'],['bezhevyj','beige'],
    ['sinij','blue'],['ser yj',null],['raznoczvetnyj',null],
    ['belyj','white'],['seryj','grey']
  ]) {
    const row=make('Лоферы FLEX HILFIGER LTH TASSEL LOAFER FM0FM05891',suffix);
    assert.equal(frColorFromCanonicalUrl(row.source_url),color);
    assert.equal(inferCatalogColor(row),color||'unknown');
  }
});

test('FR fallback cannot be tricked by lookalike hosts, querystrings, or unrelated routes', () => {
  const bad=[
    'http://frgroup.uz/ru/store/p-chernyj',
    'https://frgroup.uz.evil.example/ru/store/p-chernyj',
    'https://user:pass@frgroup.uz/ru/store/p-chernyj',
    'https://frgroup.uz:444/ru/store/p-chernyj',
    'https://frgroup.uz/ru/other/p-chernyj',
    'https://frgroup.uz/ru/store/p?search=brown',
    'https://frgroup.uz/ru/store/p-unknown',
    'https://frgroup.uz/ru/store/p-korichnevyjx',
  ];
  for (const url of bad) assert.equal(frColorFromCanonicalUrl(url),null,url);
});

test('native title color wins over supplier URL when explicitly grounded', () => {
  const row=make('Black Leather Derby','korichnevyj');
  assert.equal(inferCatalogColor(row),'black');
  assert.equal(projectCatalogItem(row).color,'black');
});

test('production catalog with no recognized merchant suffix keeps previous unknown behavior', () => {
  const row=make('Leather loafers','chernyj',{
    source_url:'https://other.example/product/black-loafer',
  });
  assert.equal(inferCatalogColor(row),'unknown');
  assert.equal(projectCatalogItem(row).color,'unknown');
});

test('actual Tommy colorways become usable in offline retrieval without pretending to verify photos', () => {
  const cases=[
    [make('Лоферы FLEX HILFIGER LTH TASSEL LOAFER','chernyj'),'loafer','black'],
    [make('Ботинки HILFIGER SDE WARM HYBRID CHUKKA','korichnevyj'),'boot','brown'],
    [make('Топсайдеры MODERN LIGHT SUEDE MOC TOE SHOE','raznoczvetnyj'),'unknown','unknown'],
  ];
  for(const [row,subtype,color] of cases){
    const item=projectCatalogItem(row);
    assert.equal(item.macroCategory,'shoes');
    assert.equal(item.subtype,subtype);
    assert.equal(item.color,color);
    assert.equal(item.visualEvidenceVerified,false);
    assert.equal(item.priceFreshness,'unknown');
  }
});
