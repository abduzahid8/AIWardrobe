import {
  catalogImageAdmission,
  hasAdmissibleCatalogImage,
  normalizeCatalogImageUrl,
} from '../../src/domain/catalogIdentity';

const malformed =
  'https://static.zara.net/assets/public/9c4f/7a2b/1d0e4c9f8a11/0f6e7d8c9b10/04395705_251_1.jpg&w=1024';

describe('catalog image URL normalization', () => {
  it('repairs the verified Zara glued-query construction without changing host or path', () => {
    expect(normalizeCatalogImageUrl(malformed)).toBe(malformed.replace('.jpg&w=1024', '.jpg?w=1024'));
    expect(normalizeCatalogImageUrl('https://cdn.test/a.jpeg&width=800&quality=80')).toBe(
      'https://cdn.test/a.jpeg?width=800&quality=80'
    );
  });

  it('leaves valid URLs, existing queries and blank values alone', () => {
    for (const value of [
      'https://cdn.test/a.jpg',
      'https://cdn.test/a.jpg?w=1024',
      'https://proxy.test/render?url=https://cdn.test/a.jpg&w=1024',
    ])
      expect(normalizeCatalogImageUrl(value)).toBe(value);
    expect(normalizeCatalogImageUrl('   ')).toBeUndefined();
    expect(normalizeCatalogImageUrl(null)).toBeUndefined();
    expect(normalizeCatalogImageUrl(42)).toBeUndefined();
  });

  it('is mirrored exactly by the Node ingestion helper', () => {
    const { normalizeCatalogImageUrl: nodeNormalize } = require('../../scripts/lib/catalogImageUrl');
    for (const value of [
      malformed,
      'https://cdn.test/a.jpeg&width=800&quality=80',
      'https://cdn.test/a.jpg',
      'https://cdn.test/a.jpg?w=1024',
      'https://proxy.test/render?url=https://cdn.test/a.jpg&w=1024',
      '   ',
      null,
      42,
    ])
      expect(nodeNormalize(value)).toBe(normalizeCatalogImageUrl(value));
  });
});

describe('catalog image admission', () => {
  it('admits and reports the repaired Zara URL instead of excluding the source', () => {
    expect(catalogImageAdmission(malformed)).toEqual({
      ok: true,
      url: malformed.replace('.jpg&w=1024', '.jpg?w=1024'),
      repaired: true,
    });
    expect(hasAdmissibleCatalogImage({ image_url: malformed })).toBe(true);
  });

  it('keeps already valid catalog images admissible', () => {
    expect(catalogImageAdmission('https://cdn.test/a.jpg?w=1024')).toEqual({
      ok: true,
      url: 'https://cdn.test/a.jpg?w=1024',
      repaired: false,
    });
    expect(hasAdmissibleCatalogImage({ imageUrl: 'https://cdn.test/b.webp' })).toBe(true);
  });

  it('refuses only structurally unusable image values', () => {
    expect(catalogImageAdmission(undefined)).toEqual({ ok: false, reason: 'image_missing' });
    expect(catalogImageAdmission('   ')).toEqual({ ok: false, reason: 'image_missing' });
    expect(catalogImageAdmission('https://cdn.test/a b.jpg')).toEqual({
      ok: false,
      reason: 'image_malformed',
    });
    expect(catalogImageAdmission('javascript:alert(1)')).toEqual({
      ok: false,
      reason: 'image_malformed',
    });
    expect(catalogImageAdmission('/products/shirt.jpg')).toEqual({
      ok: false,
      reason: 'image_malformed',
    });
    expect(catalogImageAdmission('https://cdn.test/a.jpg&w')).toEqual({
      ok: false,
      reason: 'image_malformed',
    });
    expect(catalogImageAdmission('https://cdn.test/a.jpg?w=1?x=2')).toEqual({
      ok: false,
      reason: 'image_malformed',
    });
    expect(hasAdmissibleCatalogImage(null)).toBe(false);
  });

  it('does not reject a valid query value that merely contains an image path', () => {
    const proxy = 'https://proxy.test/render?url=https://cdn.test/a.jpg&w=1024';
    expect(catalogImageAdmission(proxy)).toEqual({ ok: true, url: proxy, repaired: false });
  });
});
