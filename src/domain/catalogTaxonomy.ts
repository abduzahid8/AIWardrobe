/** Derived taxonomy only. Original IDs, source categories, prices and links remain intact. */
export interface CatalogTaxonomy {slot:string;subtype:string;productKind:'garment'|'complete-suit'}
export function sourceCatalogTaxonomy(item:{name?:string|null;type?:string|null;category?:string|null;garment_type?:string|null}):CatalogTaxonomy|undefined {
 const name=String(item.name||'').toLowerCase().replace(/[_-]+/g,' ');
 const source=String(item.garment_type||item.type||item.category||'').toLowerCase();
 if(/\bcardigans?\b|кардиган/.test(name))return {slot:'outerwear',subtype:'cardigan',productKind:'garment'};
 // Specific product nouns precede the broad "suit" and "Oxford" tokens.
 if(/\b(blazers?|suit jackets?|tailored jackets?|sports? coats?|dinner jackets?)\b|пиджак/.test(name))return {slot:'outerwear',subtype:'blazer',productKind:'garment'};
 if(/\boxfords?\b/.test(name)&&(/\bshirts?\b/.test(name)||/upper_body|tops|shirts/.test(source)))return {slot:'top',subtype:'dress-shirt',productKind:'garment'};
 if(/\b(shirts?|trousers?|pants?|skirts?|shorts?|jackets?|waistcoats?|vests?|shoes?|boots?|loafers?)\b/.test(name))return;
 if(/\bsuits?\b|\b(?:two|three|2|3) piece\b.*\b(?:set|tailor)|костюм/.test(name)
   ||/^(?:suits?|complete.suit|outfits)$/.test(source))return {slot:'outfits',subtype:'complete-suit',productKind:'complete-suit'};
 return;
}
