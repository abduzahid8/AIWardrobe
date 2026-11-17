# AI-led whole-look decisions, 17 November 2026

Goal: the actual fashion AI must decide which assembled, pre-admitted looks deserve evaluation, not a lengthening list of product-title regex prohibitions.

- Before: formula / deterministic seed ranking generally allocated the 8 scarce, paid full-photo critic slots. An excellent creative direction could disappear before the model ever saw the photographs.
- After: Creative Director plans roles first; after SKU retrieval a bounded second NVIDIA light model sees up to 28 REAL full looks with only source-hash-current confidently image-observed visual properties and anonymized L00-L27 identifiers. It chooses 6 full outfits and leaves 2 slots for alternatives.
- When AI slate selection succeeds, aesthetic/occasion regex vetoes stop deciding which combinations may be sent to the actual-photo critic; weather safety, inventory, ownership, exact price and budget, merchant/provenance, photo authorization and independently scored visual review remain.
- The visual critic reviews actual garment photographs and rejects weak looks; strict minimum style/occasion/formality scores apply. The lightweight selector never approves a card.
- If the selector is missing, fails, exceeds its time budget or nominates fewer than 3 admitted candidates, the original deterministic editorial selection is restored. No hidden unreviewed filler.
- Isolated S6 staging only, behind server-only STYLIST6_AI_SLATE_ENABLED=1, and existing Creative Director + visual flags. Do not turn on in production or client request bodies.
- Model selection uses anonymous per-request L tokens, never raw user IDs, SKU IDs, uploaded photos, merchant URLs or secrets. All visual adjectives sent to the selector are source-hash-verified attributes rather than title guesses.

Verification required: focused selector tests, full Stylist 6 backend, exact SHA READY Preview and pinned staging, authenticated iOS generations with the *actual source product photographs*, then >=30 blinded S5/S6 scenario comparisons by 3 people. CI or a successful server response is not proof of taste. Production Stylist 5 remains unchanged.

## Real-device acceptance checkpoint — 17 Nov 2026

**The objective cannot change:** a truly excellent PERSONAL STYLIST must choose three attractive, wearable, context-matching, visually different purchasable looks from real merchandise and real product photos. HTTP success or three arbitrary garment tuples are not acceptance. The independent NVIDIA photo critic must approve each displayed outfit; never invent a third.

### Authenticated iOS Simulator results, actual Render staging compute

All runs were driven through the installed AIWardrobe QA simulator and signed Supabase Render canary backed by real catalog items, NOT static fixtures. The deployed compute for these tests was 3bed08d4.

| Real user selection | Final visual receipt | Representative real merchandise |
|---|---|---|
| Classic / Work / New clothing | 3 looks; NVIDIA_APPLIED c8/v8/a0 shown3/3 | White poplin and dark trousers with black loafers; light blue Oxford with muted blazer, navy trousers and brown loafers; black shirt with houndstooth double-breasted blazer, grey trousers and dark loafers |
| Old Money / Date night / New clothing | 3 looks; NVIDIA_APPLIED c8/v8/a0 shown3/3 | Navy merino knit, chinos, burgundy loafers; brown linen polo, brown chinos, leather loafers; pale blue Oxford, warm chinos, light loafers |
| Casual / Sport / New clothing | 3 looks; NVIDIA_APPLIED c8/v8/a0 shown3/3 | Technical tee, zip jacket, sweatpants and trail trainers; performance tank, joggers and training shoes; technical long-sleeve tee, sweatpants and trail shoes |

**9/9 actually displayed cards had photo-verdict evidence.** Stock is NOT verified: each merchant card honestly says check size and stock at store. UI screenshots were inspected locally; user IDs, tokens, identifying screen content and secrets are not committed.

One separate earlier real canary run confirmed the lightweight Fashion Director had selected 6 real looks from 28 candidates with one provider call and an available unique triple. The three later 3-card iOS screen receipts prove the final photo review, but do not independently expose the AI selector receipt.

### Aesthetic assessment and remaining limitations

- Classic Work: coherent, but the black shirt and patterned double-breasted blazer are too expressive for a conservative office.
- Old Money Date: wearable, attractive color ranges, but the three sets are still mostly chinos-and-loafers with changing tops. Independence by SKU is NOT the same as editorially distinct looks.
- Casual Sport: genuine technical clothing and athletic footwear, but trail shoes are not universally suited to indoor training. Subdivide gym/trail/running based on the brief and gear.
- Prior copy occasionally mislabeled an actual technical tee as Knitwear, or named a single garment instead of the styled set. Revision 27c77fe contains grounded pair headlines and copy-only safeguards in English/Russian/Uzbek. Verify deployed SHA before calling that revision live.
- Some baskets contain several merchant currencies. The UI must never invent an unverified combined total; any combined estimate must use current, valid CBU conversion.
- Render is a currently isolated staging canary, NOT a general public production rollout. Production Stylist 5 is unchanged. Vercel reached its daily free-tier deployment cap; capacity and rollback are still release dependencies.

### Binding next actions, in order

1. Raise human-perceived beauty, utility, and actual structural diversity while keeping all source, budget, weather, user exclusion, SKU and independent-photo gates. No cosmetic test-count theater, no weak third-card fallback.
2. Test more real user scenes: cold and hot weather, explicit color exclusions, budgets and exchange rates, hybrid/owned wardrobe, occasion boundaries, stock and purchase URLs. Include independent blinded style comparisons (goal: 30 scenarios, 3 raters) before asserting professional-grade fashion quality.
3. Verify exact deployed commit and CI; address provider capacity and response times before making Stylist 6 the main production default. The dev-only canary must never become an unauthenticated production bypass.
4. Resume from THIS measured real-device acceptance state and known defects, rather than restarting architectural debates. The final outcome is a genuinely scalable personal stylist.
