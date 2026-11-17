/**
 * Stylist 6 — evidence-only finishing pass for quiet office outfits.
 *
 * A deterministic, small *tie-breaker*, not an image evaluation. All items
 * have already passed the trusted feasibility and coherence gates. The pass
 * cannot invent fabric, colors, stock or the user's preferences. Unknown or
 * conflicting catalog metadata stays neutral. Wardrobe/mixed modes and any
 * pinned anchor are deliberately out of scope.
 */
import { colorTokensOf } from '../../src/domain/outfitDiversity.ts';

export const FASHION_FINISH_VERSION = 's6-quiet-office-finish-v1';
const OFFICE = /\b(work|office|business|meeting|conference)\b|работ|офис|делов|встреч|\b(ofis|ish|biznes|uchrashuv)\b/i;
const STYLES = new Set(['classic','old_money','semi_classic','business_casual']);
const EARTH = new Set(['cream','beige','tan','brown','green']);
const DARK = new Set(['black','charcoal','navy','dark-neutral']);
const QUIET = new Set(['white','cream','beige','tan','brown','black','charcoal','navy','dark-neutral','grey','blue','light-blue','green']);
const ACCENTS = new Set(['burgundy','red','rust','pink','yellow']);
const known = value => typeof value === 'string' && value && !['unknown','neutral','missing','other'].includes(value.toLowerCase());
const round = n => Number(n.toFixed(4));

// A marketing name may contain two colors, e.g. "black/brown". Refuse to
// guess which color occupies the visible product rather than score a fiction.
function evidencedHue(item) {
  const explicit = known(item?.color) ? colorTokensOf({ color:item.color }) : [];
  const observed = explicit.length ? explicit : colorTokensOf({ name:item?.name || '' });
  return observed.length === 1 ? observed[0] : null;
}

const neutral = () => ({
  version:FASHION_FINISH_VERSION, active:false, adjustment:0,
  paletteHarmony:0, shoeAppropriateness:0, silhouetteBalance:0,
  textureCoordination:0, genericOfficePenalty:0, evidence:[],
});

export function evaluateFashionFinish({ items=[], descriptors=[], context={}, requiredIds=[] }={}) {
  if (context.sourceMode !== 'catalog' || !STYLES.has(context.style) ||
      !OFFICE.test(String(context.occasion || '')) || requiredIds.length ||
      !Array.isArray(items) || !Array.isArray(descriptors) || items.length !== descriptors.length) return neutral();

  const bySlot = Object.fromEntries(items.map((item,i)=>[
    item?.macroCategory, {item, d:descriptors[i], hue:evidencedHue(item)},
  ]).filter(([slot])=>['top','outerwear','bottom','shoes'].includes(slot)));
  const top=bySlot.top, layer=bySlot.outerwear, bottom=bySlot.bottom, shoes=bySlot.shoes;
  if (!top || !bottom || !shoes) return neutral();
  const request = String(context.request || '');
  const favoredColors = Array.isArray(context.preferences?.favoriteColors)
    ? context.preferences.favoriteColors.flatMap(color => colorTokensOf({ color:String(color) })) : [];
  const requestedShoe = /\b(sneakers?|trainers?|running shoes?|oxfords?|derb(?:y|ies)|loafers?|black shoes?)\b|кроссовк|кед|оксфорд|дерби|лофер/i.test(request);
  const evidence=[];
  let paletteHarmony=0, shoeAppropriateness=0, silhouetteBalance=0, textureCoordination=0, genericOfficePenalty=0;
  const garmentHues=[top.hue,layer?.hue,bottom.hue].filter(Boolean);
  const hues=[...garmentHues,shoes.hue].filter(Boolean);
  const garmentSet=new Set(garmentHues);
  const favorites=new Set(favoredColors);

  if (hues.length >= 3) {
    const accents = new Set(hues.filter(hue=>ACCENTS.has(hue)));
    if (accents.size>1 && ![...accents].every(hue=>favorites.has(hue))) {
      paletteHarmony -= 0.012;
      evidence.push('competing_evidenced_accents');
    } else if (hues.every(hue=>QUIET.has(hue) || accents.has(hue))) {
      paletteHarmony += 0.008;
      evidence.push('restrained_confirmed_palette');
    }
    if (shoes.hue === 'black' && garmentHues.length>=2 &&
        garmentHues.every(hue=>EARTH.has(hue) || hue==='burgundy') &&
        !favorites.has('black') && !/\bblack\b|ч[её]рн/i.test(request)) {
      paletteHarmony -= 0.019;
      evidence.push('unanchored_black_footwear_with_earth_palette');
    } else if (shoes.hue && DARK.has(shoes.hue) && [...garmentSet].some(hue=>DARK.has(hue))) {
      paletteHarmony += 0.006;
      evidence.push('dark_footwear_color_anchor');
    } else if (['brown','tan'].includes(shoes.hue) && [...garmentSet].some(hue=>EARTH.has(hue))) {
      paletteHarmony += 0.008;
      evidence.push('warm_footwear_color_anchor');
    }
  }

  const shoeSub=shoes.d?.subtype;
  const topSub=top.d?.subtype;
  const layerSub=layer?.d?.subtype;
  const bottomSub=bottom.d?.subtype;
  const tailored=['tailored-trouser','trouser'].includes(bottomSub);
  const softTop=['fine-knit','sweater','knit-polo'].includes(topSub) || layerSub==='cardigan';
  const tailoredTop=['dress-shirt','shirt'].includes(topSub) && layerSub==='blazer';
  const formalShoe=['derby','oxford','monk'].includes(shoeSub);
  const relaxedShoe=['loafer','chelsea'].includes(shoeSub);
  if (!requestedShoe) {
    if (softTop && relaxedShoe && shoes.d?.casualSole!==true) {
      shoeAppropriateness+=0.012;
      evidence.push('smart_footwear_supports_soft_tailoring');
    } else if (tailoredTop && tailored && formalShoe && shoes.d?.casualSole!==true) {
      shoeAppropriateness+=0.010;
      evidence.push('formal_footwear_supports_tailoring');
    } else if (softTop && bottomSub==='chino' && shoeSub==='oxford') {
      shoeAppropriateness-=0.010;
      evidence.push('strict_oxford_with_soft_chinos');
    }
  }
  if (tailoredTop && tailored && formalShoe) silhouetteBalance+=0.007;
  else if (softTop && ['chino','tailored-trouser','trouser'].includes(bottomSub)) silhouetteBalance+=0.008;

  // Fabric must come from source-backed descriptor evidence. Cashmere/wool
  // are not inferred from the product's price or brand.
  const garmentTexture = [top.d,layer?.d,bottom.d].some(d=>
    d && (['knit','textured'].includes(d.texture) ||
      Array.isArray(d.materials) && d.materials.some(m=>['wool','cashmere','merino','corduroy'].includes(m))));
  const shoeSuede = shoes.d && (shoes.d.texture==='suede' ||
    Array.isArray(shoes.d.materials) && shoes.d.materials.includes('suede'));
  if (softTop && garmentTexture && shoeSuede && relaxedShoe) {
    textureCoordination+=0.007;
    evidence.push('confirmed_soft_texture_and_suede');
  }

  // The basic shirt-trouser-shoe formula is valid (particularly in summer).
  // Only a low-evidence, unlayered generic office combination loses a small
  // tie against a polished layered alternative; never penalize an expressly
  // requested no-layer look or warm-weather composition.
  const warm = Number(context.weather?.temp ?? context.weather?.temperature);
  const hot = Number.isFinite(warm) && warm >= 25;
  const explicitSimple = /\b(no jacket|no blazer|no layer|just a shirt|without a jacket)\b|без пиджак|без куртк|без слоя/i.test(request);
  if (!layer && !hot && !explicitSimple && ['shirt','dress-shirt'].includes(topSub) &&
      ['trouser','tailored-trouser'].includes(bottomSub) &&
      ['loafer','derby','oxford'].includes(shoeSub) &&
      [top.d,bottom.d,shoes.d].every(d=>d && (d.pattern==='solid' || d.pattern==='unknown'))) {
    genericOfficePenalty=-0.009;
    evidence.push('plain_unlayered_office_formula_tiebreak');
  }

  const adjustment=Math.max(-0.045,Math.min(0.038,paletteHarmony+shoeAppropriateness+
    silhouetteBalance+textureCoordination+genericOfficePenalty));
  return {version:FASHION_FINISH_VERSION,active:true,adjustment:round(adjustment),
    paletteHarmony:round(paletteHarmony),shoeAppropriateness:round(shoeAppropriateness),
    silhouetteBalance:round(silhouetteBalance),textureCoordination:round(textureCoordination),
    genericOfficePenalty:round(genericOfficePenalty),evidence};
}
