import {describeGarment,type GarmentEvidence} from './garmentDescriptor.ts';
import type {StylingContext} from './stylingContext.ts';
/** The v200 mobile judge treats Work as formal and rejects knit layers. Keep
 * that wire contract for clients without the new composition capability; this
 * compatibility profile is not used by updated Personal Stylist clients. */
export function legacyClientCompatible(item:GarmentEvidence,context:StylingContext):boolean {
 if(!context.legacyFourSlot || context.dressCode!=='smart' || !/\b(work|office|business)\b|работ/.test(context.occasion))return true;
 const d=describeGarment(item),text=`${item.name||''} ${item.description||''}`;
 if(d.slot==='outerwear')return ['blazer','coat'].includes(d.subtype);
 if(d.slot==='shoes')return !d.casualSole&&!d.rugged&&!d.athletic&&['oxford','derby','monk','loafer','chelsea','dress-boot'].includes(d.subtype);
 if(d.slot==='top')return ['shirt','dress-shirt','polo','fine-knit'].includes(d.subtype)&&!/\bpiqu[eé]|\brelaxed[ -]fit\b/i.test(text);
 return true;
}
