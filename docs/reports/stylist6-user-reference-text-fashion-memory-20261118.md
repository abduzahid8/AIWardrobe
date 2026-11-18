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
