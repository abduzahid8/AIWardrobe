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
