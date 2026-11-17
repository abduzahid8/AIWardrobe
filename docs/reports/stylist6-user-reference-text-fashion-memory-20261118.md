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
