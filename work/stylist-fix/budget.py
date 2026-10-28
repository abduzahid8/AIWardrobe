from pathlib import Path
p=Path('backend/supabase/functions/generate-outfits/index.ts');s=p.read_text().replace('`${slot}:${canonicalProductKey(item)}`','`${slot}:${isShopRow(item) ? canonicalProductKey(item) : item.id}`');s=s.replace('    budgetFx?: BudgetFxContext,\n): Promise<any[]> {','    budgetFx?: BudgetFxContext,\n    nativeBudget?: {amount: number; currency: string},\n): Promise<any[]> {',1);s=s.replace("                .abortSignal(controller.signal)\n            let slotFilter", "                .abortSignal(controller.signal)\n            if (nativeBudget) q = q.eq('currency', nativeBudget.currency).lte('price', nativeBudget.amount).order('price', {ascending:true})\n            let slotFilter",1)
s=s.replace('''                    parsedBudget.fx = await loadRequestFx()
                    preferences.maxShopBudget = parsedBudget''','''                    try { parsedBudget.fx = await loadRequestFx() }
                    catch (error) {
                        if (body.preferences?.budgetRatesId || new Set(budgetCandidatesFromTextEdge(`${prompt} ${occasion}`).map(b => b.currency)).size > 1) throw error
                        // Same-currency prices require no exchange rate. Exclude
                        // foreign quotes rather than inventing a conversion.
                        console.warn('[budget-rates] using_native_currency_only', {currency: parsedBudget.currency})
                    }
                    preferences.maxShopBudget = parsedBudget''',1)
s=s.replace('shopEligible, parsedBudget ? requestFx : undefined)', 'shopEligible, parsedBudget ? requestFx : undefined, parsedBudget && !requestFx ? parsedBudget : undefined)',1)
s=s.replace('''            await loadRequestFx()
            parsedBudget.fx = requestFx
            preferences.maxShopBudget = parsedBudget''','''            try { await loadRequestFx() }
            catch (error) {
                if (body.preferences?.budgetRatesId) throw error
                wardrobeItems = wardrobeItems.filter((item: any) => !isShopPricedItem(item) || String(item.currency || '').toUpperCase() === parsedBudget!.currency)
                console.warn('[budget-rates] using_native_currency_only', {currency: parsedBudget.currency})
            }
            parsedBudget.fx = requestFx
            preferences.maxShopBudget = parsedBudget''',1)
p.write_text(s)
p=Path('screens/AIOutfitmaker.tsx');s=p.read_text();s=s.replace("catch { throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable'); }", "catch {\n          if (mixedLimits) throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable');\n          // Let the server find catalog prices in the requested currency.\n          // No conversion or guessed reference is used in this mode.\n        }");s=s.replace('shopBudget = parseBudgetFromText(promptText, fx);','shopBudget = parseBudgetFromText(promptText, fx);');p.write_text(s)
