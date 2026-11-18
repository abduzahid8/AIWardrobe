# Stylist 6 — human-labeled TEXT reference memory for AI fashion concepts

## Motivation: actual native QA, not ranker benchmarks

Pinned real paid staging `af5e8ed`, `5f96aa7`, authenticated iPhone
New clothing / Classic / Work and Date night: two well-formed 4-SKU
photoset outfits repeatedly appeared on the device, but BOTH were
near-clone shirt+blazer or shirt/henley+cardigan architectures.
Under `5f96aa7`, the SECOND independent 120B compositor genuinely
saw `unseen_source_skus` from the wider authenticated ~400-product
catalogue, yet all THREE additional real-photo whole looks were
rejected on Date night. Mere SKU recall is not fashion expertise.

## Source of creative inspiration

The project already contains 46 **user-supplied, manually annotated
flat-lay fashion reference** descriptions in
`benchmarks/stylist6/references/image_teaste_46.json`.
Their `rights_status` is NOT verified for these photos.
The labels are approximate aesthetic observations, not a blind human
quality assessment or material/fit/availability proof.
We do NOT upload, expose, download, transmit, claim rights in,
or even read those original JPEGs for this feature.

New `backend/stylist6/editorial_text_moodboards.mjs` retrieves
up to three different human-labeled whole-look TEXT ideas with the
requested style and occasion (e.g. true Classic Work, true Classic
Date night). It deliberately diversifies coat/cardigan/overshirt/
blazer family by the actual source-caption garment relationships.
It omits raw image names, user IDs, source-account, media SHA, URLs,
unmatched occasions, and any palette/garment descriptors matching
explicitly excluded colors. It never fabricates a real SKU.

The 30B independent Creative Director receives these *optional,
descriptive* wardrobe references BEFORE its 3 first-look concepts.
The 120B exact-SKU composer also receives the same optional style
moodboard when composing its six real-shopping ensembles, including
the artist's second attempt if the PHOTO editor complained.
The model is told the archive includes **TEXT ONLY**, not inspected
photographs, not verified outfit quality, and never an instruction
to replicate impossible merchandise. Its source-bound item menus,
owner limits, weather, shop URL, currency, budgets and stock
precede all inspiration. Actual full-look independently NVIDIA-
reviewed original source PRODUCT photos still control approval;
the final visual editor may return only one or two when appropriate.

## Test and no-go

New tests prove: 3 different user-caption outerwear directions for
Classic Work and Date night; original user reference file identities,
unlicensed media and source-account metadata never leave the server;
explicit disliked colors are respected; wrong occasion returns no
inspiration; real NVIDIA prompt receives the ideas without revealing
garment SKU IDs, merchant media URLs or private account data. Existing
full gates and source/auth/physical/photo/merchant checks remain.

This is a **fashion-context intervention**, not evidence of achieved
human-caliber fashion. Next real mobile QA must establish a genuinely
new elegant ensemble or measurable improvement; do not infer success
from merely loading the 46-reference library. Human blinded 70%
S6 vs S5 / >=30 held-out scenarios / 3 judges remains NOT DONE.

## Real iPhone Classic/Work failure — 2026-11-18 07:33 Asia/Tashkent

Pinned `7e08c05`, actual authenticated iPhone Classic/New clothing/
Work generation returned `stylist6_visual_quality_unmet` with an
honest red error, **zero user-visible outfits**, and no unapproved
fallback. Supabase function `aiw-stylist6-node-staging` returned HTTP
502 at 2026-11-18 02:33:46Z. The Render process cold-started at
02:33:15–02:33:17Z. This confirms that textual owner lookbook ideas
do NOT solve creative quality by themselves.

Root QA visibility defect: staging logged the sanitized
`stylist6_safe_visual_quality` ONLY on successful results, never on
the 502 branch where all proposals were rejected. A new server-only
failure trace mirrors the existing allow-listed bounded counts/reason
enums/phase durations plus `failedReason`, to expose whether full
outfits failed by occasion, coherence or quality. It never logs
account information, product IDs, prices, model prompts, photos,
garment names or merchant URLs. Failure still refunds its original
quota and never stores/display an unverified deck. The exact photo
veto floors and budget are unchanged.

This visibility fix alone is NOT fashion-quality acceptance; review
the next actual iPhone rejection mix and fix the model-led creative
decision, not the veto threshold.

## Follow-up 2026-11-18 — five REAL Work proposals, ZERO photo approvals

Pinned \`080e0fc\`, real authenticated iPhone Classic / New clothing /
Work, Render at **2026-11-18 02:37:07Z**:
\`skuCompositionStatus=composed\`, proposed/admitted 5 real
four-item looks, independent actual-JPEG NVIDIA reviewer attempted
5, **approved ZERO**. Safe original-photo reason enums include
\`below_quality_floor\` and \`model_veto\`. No personal data or
merchant details logged. The iPhone showed
\`stylist6_visual_quality_unmet\` without a poor substitute.
~37.9 seconds server-side; independent photo judge cost ~28.2 sec.

This confirms the real ROOT issue is not client transport, the
number of prebuilt candidate SKUs, a missing output card, or an
overzealous code-title regex: the WHOLE outfits produced by the
autonomous AI are genuinely rejected by an independent photo critic.

## Architectural response: a 120B art director and explicit creative INTENT

The previous 30B Creative Director emitted only \`title\`,
\`silhouette\` from three simplistic enum values, \`palette\`,
and garment \`roles\`. Its rich actual artistic concept was
NEVER represented in output. Worse,
\`validateDirectorDirections\` discarded any two ideas sharing the
same garment CATEGORY role tuple regardless of fit, drape,
texture, construction or color story. The downstream 120B
merchant-SKU composer was asked to choose six original outfits
from a small P-token set, but saw no substantive AI fashion
ART DIRECTION from the concept stage.

On the isolated server-owned >=70s S6 budget only:
- Use the existing NVIDIA 120B model for one **professional
  concept design** call instead of the 30B lightning planner.
  Preserve the original 30B path on normal 50s preview.
- Add mandatory \`artDirection\` to the exact inventory-grounded
  model JSON schema: original 90–230-character garment
  proportions/material interaction/footwear-color-vibe statement
  for EACH of the three different outfit concepts. No merchant
  or SKU claims, no URLs, no unlicensed photos.
- Copy the bounded, URL/email-sanitized art direction as
  *untrusted descriptive DATA* into the existing 120B
  source SKU composer, enabling it to translate an actual
  designer's considered concept into current merchant products.
- Reject only genuinely duplicate designs; don't collapse
  radically different outfit concepts merely because both
  use a blazer, shirt, trousers and shoes.
- Even 120B designs must pass all unchanged hard
  source/merchant/weather/budget/owner exclusions, an
  independent actual-photo critic and the independent final
  portfolio editor. Zero approvals => honest no-look error.

Tests exercise separate 30B and 120B model caches, output schema,
private-data sanitation and two different artistic directions
built from the same nominal garment categories. Passing tests
are NOT evidence of improved real photo acceptance.

Human blind >=70% preference over S5 on >=30 held-out
situations with 3 independent reviewers remains NOT DONE.

## Independent cross-model VISUAL fashion curator, 2026-11-18

Actual pinned \`a1dfc12\`, native iPhone Classic / Date night at
02:46:36Z: **7/8 PHOTO approvals, three complete real four-SKU looks
shown in the actual iOS client, 39.65s including recovery storage**.
All three use near-identical cardigan + chinos architecture, with
variations in shirts, colors and shoes. Nemotron 120B as creative
director, exact-source SKU designer, independent single-outfit
photo critic, and final PHOTO collection curator shares the SAME
NVIDIA/Nemotron model family. This appears to reinforce the same
aesthetic bias across the entire review chain. Three verified photos
are not three convincing independent styles.

An authenticated source-bound NVIDIA model-directory check exposed
\`moonshotai/kimi-k3.1-pro\`; its DIFFERENT-LAB multimodal endpoint
replied to an actual generated JPEG probe with valid JSON in ~1.3s.
The alternative Qwen 3.5 endpoint returned 410 and Mistral/Gemma
timeout at 14s; therefore Kimi is the independently-tested
alternative without inventing a new API key, vendor or capability.

On ISOLATED 72s Stylist6 staging ONLY, when >=3 independently
photo-approved, source-verified looks form at least one feasible
triple, send the SAME original, already-authorized, actual product
photo contact sheet to Nemotron 120B fashion editor and an anonymous
Kimi vision curator CONCURRENTLY (not serial 2x latency). Each sees
the real photographic garments but NOT the other model's decision.
Both may choose only from the SAME small server-verified entire
portfolio F-token menu. Kimi may VETO a Nemotron-selected triple
or pair by choosing an exact STRICT SUBSET only, when that subset
already passed the original all-SKU, photo, diversity, merchant,
budget and source feasibility. It may NEVER introduce an outfit,
replace a token, mix products, override a photo veto or pad a deck.
A disagreement not forming a subset is safely noted but does NOT
change the original Nemotron decision. Kimi unavailable/timeout/410
leaves the original independent PHOTO-approved deck untouched.
One source JPEG collage, no second unlicensed merchant download
wave. The original source replay and full quality floor remain
authoritative.

New diagnostic \`photoEditorialIndependentJury\` is one safe enum:
\`not_run | agreed | narrowed | disagreed | unavailable\`; the
FINAL re-edit receipt overrides the initial edit receipt when a new
photo-approved SKU wave succeeds. No private model prompts, names,
photos, catalog prices, sources or raw URL text in Render logs.

This does NOT establish aesthetic superiority: an experimental
Kimi swatch probe using textual descriptions of three cardigan
ensembles also voted all three as \"different\", so real source-JPEG
and held-out blind fashion judgments are crucial. A cross-model jury
is meaningful only if it actually rejects repetitiveness, not just
repeats the first model's permissive decisions.

## First cross-lab native run: PHOTO montage missed its deadline

Pinned \`aadab9a\`, real iPhone / Render staging Classic / Date night
at 2026-11-18 02:58:17 UTC: 6 genuine source-validated four-item
proposals, 5/6 original-photo quality approvals, 3 complete source
looks displayed, stored with \`recoverable:true\` in 34.5s.
However \`photoEditorialVisualMedia:deadline\`,
\`photoEditorialVisualJury:image_unavailable\` and
\`photoEditorialIndependentJury:not_run\`. The actual photographic
collection board did NOT finish its 3.8-second window; no
different-model counterjuror saw original photos. The 30B text
fallback selected the three-look portfolio. Thus this run proves
delivery and no source-photo laundering, NOT the new Kimi jury.

The old collage implementation separately JPEG-encoded every
photographic cell, decoded each result again, and composited
four + N different full-size transparent SVG text layers for the
same public slot/title/token text. On a constrained shared Render
CPU this is expensive even with warmed original JPEGs.
\`photo_editor_contact_sheet.mjs\` now decodes/resizes each original
licensed SKU photograph **once** to raw RGB, composites the 4-slot
grid once and renders ALL fixed labels in ONE SVG. The exact
original-source URL authorization check, strict missing-photo veto,
image-size bound, 2–8 look range and fixed 3.8-second deadline are
preserved. 32 individual real synthetic-input JPEGs for an
8-look four-item board completed in a local instrumented test in
well under 1 second, but live Render/CDN timing is NOT guaranteed.

Next e2e native result must report
\`photoEditorialVisualMedia:ready\` and independent jury
\`agreed|narrowed|disagreed|unavailable\` (not \`not_run\`)
for a feasible photographed triple before we claim any real Kimi
visual evaluation. We still cannot claim excellent diverse style.

## Exact Kimi HTTP 400 root-cause (reproduced with real provider)

On \`2ecbce1\` actual iPhone Date night run 2026-11-18
03:05:24Z, the optimized source-photo contact sheet succeeded
(\`photoEditorialVisualMedia=ready\`) and the Nemotron original-photo
portfolio jury was applied, with 3 full source-photographed looks
already approved and one feasible three-look portfolio.
The independent Kimi status was \`unavailable\`, so the user was
shown two original-source PHOTO-approved sets (never padded).

A safe isolated real NVIDIA \`/v1/chat/completions\` original-image
probe, same exact production Kimi invocation options, reproduced
HTTP 400 with this response:

    Validation: top_p is fixed at 0.95 for this model;
    received 0.9. Omit top_p or set it to 0.95.

Kimi uses provider-fixed nucleus sampling. Our shared Nemotron
request body incorrectly supplied \`top_p:0.9\` to independent Kimi,
but Nemotron accepts that option. This Kimi-specific top_p field
is now OMITTED, preserving its API-fixed 0.95 value. No new
safety threshold or fallback changes. A regression test asserts
the absence of top_p on the independent Kimi request; its same
actual original-image F-token portfolio menu, strict source
validation and no-fabrication properties still apply.

A new LIVE authenticated iPhone call must prove
\`photoEditorialIndependentJury\` becomes one of
\`agreed|narrowed|disagreed\`, not \`unavailable\` or \`not_run\`,
before counting this as a running independent reviewer.
