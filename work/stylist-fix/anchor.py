from pathlib import Path
p=Path('backend/supabase/functions/_shared/occasionRules.ts');s=p.read_text().replace('export interface JudgeableItem {','export interface JudgeableItem {\n    id?: string;');s=s.replace('''    weather?: ClimateWeather | null,
): string | null {
    if (!items || items.length === 0) return 'empty outfit';''','''    weather?: ClimateWeather | null,
    lockedItemId?: string | null,
): string | null {
    if (!items || items.length === 0) return 'empty outfit';''',1)
s=s.replace('''        const r = rejectForStyleOccasion(it, styleId, occasionId);
        if (r !== null)''','''        // The locked piece overrides aesthetic taste, never dress code, activity or weather.
        const r = rejectForStyleOccasion(it, lockedItemId && String(it.id) === lockedItemId ? 'casual' : styleId, occasionId);
        if (r !== null)''',1)
s=s.replace('''    weather?: ClimateWeather | null,
): boolean {
    return styleJudgeReason(items, styleId, occasionId, weather) === null;''','''    weather?: ClimateWeather | null,
    lockedItemId?: string | null,
): boolean {
    return styleJudgeReason(items, styleId, occasionId, weather, lockedItemId) === null;''',1);p.write_text(s)
p=Path('backend/supabase/functions/generate-outfits/index.ts');s=p.read_text().replace('''    const targetChoicesPerSlot = manualSelection
        ? 1
        : (limit >= 10 ? 16 : 12)''','''    const targetChoicesPerSlot = manualSelection
        ? 1
        : (limit >= 10 ? 16 : items.some(item => !isShopRow(item)) ? 6 : 12)''',1)
s=s.replace('rejectForStyleOccasion(item, selectionStyle, `${occasion} ${prompt}`)', "rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || '') ? 'casual' : selectionStyle, `${occasion} ${prompt}`)")
s=s.replace('const rankEligible = (item: any) => eligibleCandidate(item) && tasteEligible(item)', "const rankEligible = (item: any) => eligibleCandidate(item) && (String(item.id) === String(anchorItemId || '') || tasteEligible(item))")
s=s.replace('styleJudgeReason(resolved, style, occasion, weather)', 'styleJudgeReason(resolved, style, occasion, weather, preferences?.lockedAnchorId)')
s=s.replace('styleJudgeReason(parts, styleKey, `${occasion} ${prompt}`, weather)', 'styleJudgeReason(parts, styleKey, `${occasion} ${prompt}`, weather, anchorItemId)')
s=s.replace('styleJudgeReason(outfit.items || [], styleKey, `${occasion || \'\'} ${prompt || \'\'}`, weather)', 'styleJudgeReason(outfit.items || [], styleKey, `${occasion || \'\'} ${prompt || \'\'}`, weather, anchorItemId)')
s=s.replace('''    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const blob = `${it.name || ''} ${src.name || ''} ${src.description || ''} ${it.type || ''} ${src.type || ''}`.toLowerCase()
        for (const kw of rejectKeywords)''','''    for (const it of items) {
        if (preferences?.lockedAnchorId && String(it.id) === preferences.lockedAnchorId) continue
        const src = itemMap.get(it.id) || {}
        const blob = `${it.name || ''} ${src.name || ''} ${src.description || ''} ${it.type || ''} ${src.type || ''}`.toLowerCase()
        for (const kw of rejectKeywords)''',1)
s=s.replace('''        useProvidedWardrobeOnly = false,
        anchorItemId,''','''        useProvidedWardrobeOnly = false,
        catalogOnly = false,
        anchorItemId,''',1)
s=s.replace('''    weather = resolveStylistWeather(prompt, weather)''','''    if (catalogOnly) {
        legacyWardrobe = []
        useProvidedWardrobeOnly = true
        wardrobeOnly = false
    }
    weather = resolveStylistWeather(prompt, weather)''',1);p.write_text(s)
p=Path('backend/supabase/functions/_shared/outfitGenerationRequest.ts');s=p.read_text().replace("['wardrobeOnly', 'useProvidedWardrobeOnly']", "['wardrobeOnly', 'useProvidedWardrobeOnly', 'catalogOnly']");p.write_text(s)
p=Path('screens/AIOutfitmaker.tsx');s=p.read_text().replace('wardrobeItems: payloadItems,', "wardrobeItems: effectiveSource === 'shop' ? [] : payloadItems,\n          catalogOnly: effectiveSource === 'shop',",1)
s=s.replace("resolvedWeather);\n          const baseScore", "resolvedWeather, baseItemId);\n          const baseScore",1)
s=s.replace("resolvedWeather),\n        );", "resolvedWeather, baseItemId),\n        );",1)
a=s.index('      if (payloadItems.length === 0) {\n        if (isMounted.current) {\n          setError(t(\'outfitMaker.noShopItemsError\'')
b=s.index('\n    } else {',a);s=s[:a]+s[b:]
p.write_text(s)
