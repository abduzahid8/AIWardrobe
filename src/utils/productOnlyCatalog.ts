/**
 * Catalog rows are eligible for product discovery only when their ingestion
 * provenance explicitly promises a product-only image. This prevents a new
 * generic retailer feed from silently introducing photos on people or
 * mannequins into the shopping and outfit-generation surfaces.
 */
const PRODUCT_ONLY_PROVENANCE = /(?:no[-\s]?model|no[-\s]?mannequin|product[-\s]?only|plain[-\s]?background|white[-\s]?(?:bg|background)|elegant-menswear-2026)/i;
const VISIBLE_PERSON_EVIDENCE = /(?:\b(?:photo|image)\s+(?:shows?|features?)\b[^.]*\b(?:model|mannequin)\b|\bgarment\s+worn\s+by\s+(?:a\s+)?(?:male\s+)?model\b|\bmodel\s+(?:styled|wearing|wears)\b)/i;

export interface ProductOnlyCatalogEvidence {
    source?: string | null;
    description?: string | null;
    name?: string | null;
}

export function hasProductOnlyCatalogProvenance(row: ProductOnlyCatalogEvidence | null | undefined): boolean {
    if (!row) return false;
    // A feed label can legitimately contain "no-model" and a description
    // can say "model designed" without depicting one. Reject only explicit
    // evidence that the product image itself contains a person/mannequin.
    if (VISIBLE_PERSON_EVIDENCE.test(`${row.name ?? ''} ${row.description ?? ''}`)) return false;
    return PRODUCT_ONLY_PROVENANCE.test(`${row.source ?? ''} ${row.description ?? ''}`);
}
