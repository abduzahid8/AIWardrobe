# Stylist 6 — Visual-Editor → Creative Director feedback → new photo judgment

2026-11-17, PR #15, isolated 72s catalog staging ONLY.

## Why this is necessary

Pinned real iOS native Classic/Work and Classic/Date night on 120B
`4cb6aa2`: the independent whole-look NVIDIA critic passed 3/9 or 3/6
respectively; final actual-photo portfolio VLM selected only TWO looks
each time. These verified choices were visually near-clones: office
white/pale-blue shirt + dark blazer + trouser + formal shoes, and
date knit/polo + cardigan + trousers + Chelsea boot. They are not
different fashion concepts despite successful model/provider calls.

Previously the final editor's creative opinion was END OF PIPELINE:
only select F-token, return 1/2/3. When it deemed three valid individual
looks repetitive, it could discard one but could not create a new idea.

## Actual architectural change

1. Independent final PHOTO portfolio VLM sees a permitted original-JPEG
   contact sheet of 3+ genuinely photo-approved source looks. It still
   selects ONLY one server-proven feasible F-token. The same JSON can
   supply one **missingFashionDirection**: the observed collection's
   absent silhouette/texture/shoe/proportion contrast, not a product.
2. The server strips URLs, e-mails, raw identifiers, prompt-control
   characters and long text. Creative feedback is untrusted style DATA.
   A text-only, missing-image or failed final editor cannot supply
   this photo-grounded feedback.
3. **Only when the server owns >=22.5s remaining, a real chosen
   collection has fewer than three looks, a photo VLM actually produced
   a creative gap, catalog is source-backed and there is paid photo
   budget left**: ask the independent AI composer for an alternative
   direction from UNUSED actual original SKU tokens. Approved garment
   IDs from the first complete photographs are excluded from new
   source options.
4. Independently run the ORIGINAL physical/price/FX/merchant/size/weather
   hard gate and source truth. Only genuinely feasible 4-garment
   combinations may receive a second **real original-photo critic**
   review. Total photo critic requests are bounded to the existing
   eight baseline + at most THREE extra for this stage.
5. Re-enumerate only mathematically source-SKU-unique, individually
   NVIDIA-photo-approved feasible 1/2/3-look portfolios. If enough
   owned time remains, the actual visual portfolio editor chooses
   again, using the source JPEG board. The output must contain at
   least one NEW, independently approved outfit. Exact SOURCE/photo
   replay is mandatory before display.
6. Every timeout, quota limit, missing photo permission, malformed
   F-token, hard-gate rejection, negative second photo verdict or
   provider outage leaves the original already-approved 1–2 looks
   untouched. NO weak third card. No extra end-to-end timeout budget,
   no owner quota bypass, no unlicensed media. Production Stylist5
   and 50s preview are unchanged.

Sanitized diagnostic properties:
`photoEditorialCreativeRevision`: not_run | attempted | applied |
insufficient_safe_capacity, and bounded counts
`photoEditorialCreativeRevisionAttempted` /
`photoEditorialCreativeRevisionApproved`. No source identities,
raw editor text or personal photos are emitted in logs.

## Tests and acceptance

New regression suites prove:
- The PHOTO editor's original decision is exact F-token verified;
  the missing direction is redacted/length-limited, never a SKU.
- The visual gap actually reaches a second independent source-SKU
  designer, sends only UNUSED physical product IDs, triggers a SECOND
  paid photo critic pass, and requires a SECOND verified photo editor
  portfolio selection before showing a different ensemble.
- Invalid/timeout second composer preserves original approved deck.
- New outfit rejected by independent PHOTO critic cannot enter final
  wardrobe regardless of fashionable prose; no unreviewed padding.
- Full test suite and portable Node/Render build mandatory before
  real native staging. A LIVE 120B Work / Date night repeated result
  is STILL REQUIRED; no human blind acceptance has been run.

Do not claim the world-class personal stylist goal until >=70% real
human preference over S5 across >=30 held-out cases, three judges.
