from pathlib import Path
p=Path('backend/supabase/functions/_shared/outfitComposition.ts');s=p.read_text();a=s.index('/**');b=s.index('\nexport function needsLayering',a);s=s[:a]+'''/** Explicit composition intent takes precedence over the usual four-slot deck. */
export function resolveOutfitComposition(
  _style: string | null | undefined,
  weather?: CompositionWeather | null,
  prompt?: string | null,
): OutfitCompositionPolicy {
  const noLayer = /\\b(no|without|skip|exclude)\\s+(?:a\\s+|any\\s+)?(?:jackets?|coats?|layers?|outerwear)\\b|без\\s+(?:куртк|пиджак|пальто)/i.test(prompt || '');
  const layered = !noLayer;
  return { layered, activeSlots: layered ? [...LAYERED_SLOTS] : ['top', 'bottom', 'shoes'] };
}

/** A destination temperature in the request overrides the current local weather. */
export function resolveStylistWeather(prompt?: string | null, weather?: CompositionWeather | null): CompositionWeather | undefined {
  const text = prompt || '';
  const explicit = text.match(/(-?\\d{1,2}(?:\\.\\d+)?)\\s*(?:°\\s*C|degrees?\\s*(?:C|celsius)|celsius)\\b/i);
  if (explicit) return {temp: Number(explicit[1]), condition: 'requested temperature'};
  if (/\\b(freezing|snow|winter|sub[ -]?zero)\\b|мороз|зим[ауы]/i.test(text)) return {temp: 0, condition: 'winter'};
  if (/\\bsummer\\b|лет[оа]/i.test(text)) return {temp: 30, condition: 'summer'};
  return weather || undefined;
}
''' +s[b:];p.write_text(s)
p=Path('backend/supabase/functions/generate-outfits/index.ts');s=p.read_text().replace('needsLayering as needsLayeringShared }','needsLayering as needsLayeringShared, resolveStylistWeather }');s=s.replace("    const selectionStyle =", "    weather = resolveStylistWeather(prompt, weather)\n    const selectionStyle =",1)
# Fetch the exact owned anchor through the user's RLS client even in catalog mode.
needle='''        // ── 3b. Attach inferred style metadata''';a=s.index(needle);s=s[:a]+'''        if (anchorItemId && useProvidedWardrobeOnly && !wardrobeItems.some((item: any) => String(item.id) === String(anchorItemId))) {
            const {data: anchor, error: anchorError} = await userSupabase.from('clothing_items')
                .select('id,name,type,category,primary_color,brand,description,image_url')
                .eq('id', anchorItemId).eq('user_id', requestUserId).maybeSingle()
            if (anchorError) throw new CatalogLookupError()
            if (!anchor) return new Response(JSON.stringify({success:false,outfits:[],error:'anchor_not_found'}), {status:422,headers:{...corsHeaders,'Content-Type':'application/json'}})
            wardrobeItems.push({...anchor, macroCategory: macroCategory(anchor.category || '', anchor.type || '', anchor.name || ''),
                color:anchor.primary_color || 'neutral', imageUrl:anchor.image_url, isWardrobe:true, isShopItem:false})
        }

''' +s[a:]
# Explicit anchor wins over legacy forced-shop policy.
s=s.replace('!isShopOnlyDeck(stylePreferences, occasion)', '(!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId)')
s=s.replace('if (isShopOnlyDeck(stylePreferences, occasion)) {', 'if (isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId) {')
s=s.replace('const shopOnly = isShopOnlyDeck(stylePreferences, occasion)', 'const shopOnly = isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId')
# Catalog contains misclassified footwear; search product names as well as labels.
s=s.replace("slotFilter = 'category.eq.shoes,garment_type.eq.shoes'", "slotFilter = 'category.eq.shoes,garment_type.eq.shoes,name.ilike.%shoe%,name.ilike.%boot%,name.ilike.%loafer%,name.ilike.%trainer%,name.ilike.%sneaker%,name.ilike.%derby%,name.ilike.%espadrill%,name.ilike.%moccasin%'")
s=s.replace('''        countBySlot.set(slot, (countBySlot.get(slot) || 0) + 1)''','''        const key = `${slot}:${canonicalProductKey(item)}`
        if (seenProducts.has(key)) continue
        seenProducts.add(key)
        countBySlot.set(slot, (countBySlot.get(slot) || 0) + 1)''',1);s=s.replace('''    const countBySlot = new Map<string, number>()

    for (const item of items)''','''    const countBySlot = new Map<string, number>()
    const seenProducts = new Set<string>()

    for (const item of items)''',1)
# Insulated clothing and formal-intent rules must be applied before shortlist truncation.
s=s.replace('&& !rejectForClimate(item, weather) && !rejectForActivity', '&& !rejectForClimate(item, weather) && !rejectForRequestIntent(item, `${occasion} ${prompt}`, weather) && !rejectForActivity')
s=s.replace('rejectForClimate, rejectForActivity }', 'rejectForClimate, rejectForRequestIntent, rejectForActivity }')
# Stop silently exempting the anchor from the final weather/occasion judge.
a=s.index('            // The locked anchor is user-chosen:');b=s.index('            if (reason || !complete)',a)
s=s[:a]+'''            const hasAnchor = !anchorItemId || (outfit.items || []).some((it: any) => String(it.id) === String(anchorItemId))
            const reason = hasAnchor ? styleJudgeReason(outfit.items || [], styleKey, `${occasion || ''} ${prompt || ''}`, weather) : 'anchor_missing'
            const complete = !fourSlotOutfitReason(outfit.items, layered)
''' +s[b:]
# Log decisions rather than full user prompts.
s=s.replace('console.log(`Outfit gen: ${wardrobeItems.length} items, prompt="${prompt}", style=${stylePreferences}`)', 'console.log(\'[generation-pool]\', {count: wardrobeItems.length, style: selectionStyle, slots: Object.fromEntries([\'top\',\'bottom\',\'shoes\',\'outerwear\'].map(slot => [slot, wardrobeItems.filter((item: any) => item.macroCategory === slot).length]))})')
s=s.replace('''            if (missingSlots.length) return new Response''','''            if (missingSlots.length) console.warn('[generation-reject]', {code:'no_matching_clothing', missingSlots, wardrobeOnly, useProvidedWardrobeOnly})
            if (missingSlots.length) return new Response''',1)
p.write_text(s)
p=Path('backend/supabase/functions/_shared/occasionRules.ts');s=p.read_text();a=s.index('export type OccasionFormality');s=s[:a]+'''/** Admit scarce required pieces before ranking, rather than discovering them after truncation. */
export function rejectForRequestIntent(item: JudgeableItem, request: string, weather?: ClimateWeather | null): string | null {
    const slot = macroOf(item), text = blobOf(item);
    if (/\\b(no|without|skip|exclude)\\s+(?:a\\s+|any\\s+)?(?:jackets?|coats?|layers?|outerwear)\\b|без\\s+(?:куртк|пиджак|пальто)/i.test(request) && slot === 'outerwear') return 'requested no outerwear';
    if (weather?.temp != null && weather.temp < 5) {
        if (slot === 'outerwear' && !/\\b(puffer|parka|down|insulated|overcoat|topcoat|winter coat|shearling|quilted)\\b/.test(text)) return 'cold request requires insulated coat';
        if (slot === 'shoes' && !/\\b(boots?|chelsea|winter|insulated|snow)\\b/.test(text)) return 'cold request requires winter boots';
    }
    if (/black[ -]?tie|блэк[ -]?тай|смокинг/i.test(request)) {
        if (slot === 'outerwear' && !/tuxedo|dinner jacket/.test(text)) return 'black tie requires dinner jacket';
        if (slot === 'bottom' && !/tuxedo|dinner suit|black trousers|black pants|formal trousers/.test(text)) return 'black tie requires evening trousers';
        if (slot === 'top' && !/tuxedo|dress shirt|formal shirt|white shirt/.test(text)) return 'black tie requires formal shirt';
        if (slot === 'shoes' && (!/black/.test(text) || !/patent|oxford|opera pump/.test(text) || /brown|suede|loafer/.test(text))) return 'black tie requires black evening shoes';
    }
    return null;
}

''' +s[a:];s=s.replace('/tuxedo|dinner jacket|black suit|black blazer|black jacket/', '/tuxedo|dinner jacket/');p.write_text(s)
p=Path('screens/AIOutfitmaker.tsx');s=p.read_text();s="import { resolveStylistWeather } from '../backend/supabase/functions/_shared/outfitComposition';\n"+s;s=s.replace('const resolvedWeather = weather ?? undefined;', 'const resolvedWeather = resolveStylistWeather(promptText, weather);');s=s.replace('isFourSlotLook(outfit.items) || isCompleteNormalizedOutfit(outfit.items)', 'isFourSlotLook(outfit.items, data.layered !== false)');s=s.replace("const clientDiversitySlots: DiversitySlot[] = ['top', 'outerwear', 'bottom', 'shoes'];", "const clientDiversitySlots: DiversitySlot[] = data.layered === false ? ['top', 'bottom', 'shoes'] : ['top', 'outerwear', 'bottom', 'shoes'];");p.write_text(s)
p=Path('features/outfit-generator/utils/normalizeSlots.ts');s=p.read_text().replace('export const isCompleteNormalizedOutfit = (items: any[]) => fourSlotOutfitReason(items) === null;', 'export const isCompleteNormalizedOutfit = (items: any[], requireLayer = true) => fourSlotOutfitReason(items, requireLayer) === null;').replace('export function isFourSlotLook(items: any[] | null | undefined): boolean {\n  return fourSlotOutfitReason(items) === null;', 'export function isFourSlotLook(items: any[] | null | undefined, requireLayer = true): boolean {\n  return fourSlotOutfitReason(items, requireLayer) === null;');p.write_text(s)
