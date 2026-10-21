# Inline wardrobe media migration

Status: fixture-verified preparation; existing customer rows have not been migrated.

The current sync pipeline converts pending inline images to private references before writing records. Current iOS component probes cover rendering, identity clearing and recovery after published refreshed credentials; Android native callback probes now cover private loading and refreshed-token recovery after the header-forwarding repair; full installed-app flows, Android visual clearing and compatibility with released clients still require verification. Do not infer released-client compatibility from source maps or Node service tests. Keep the existing customer references until those deployment requirements are established.

The maintenance module is `scripts/maintenance/inline-wardrobe-media.mjs`. It accepts one explicitly supplied row and has no bulk scan/apply command. Use a service-role SDK only in the administrative process; keep credentials, image bytes, customer row identifiers and operational receipts out of source control and logs.

Before rollout:

1. Verify owner-authenticated native image rendering on the supported installed builds, including cold start, token refresh and account switching. Identify how unsupported released clients are handled before changing their references.
2. Inspect the bucket's privacy/MIME/size constraints and the `set_clothing_items_updated_at` trigger. The conditional update relies on the database maintaining that timestamp for every edit.
3. Audit candidate formats and sizes without exporting image payloads. Current aggregate inspection found 17 JPEG and 1 PNG inline references; this does not prove that every payload decodes correctly.
4. Run the fixture suite and confirm its object/account cleanup. Record pre-migration counts and metadata digests for the actual authorized batch.

For each authorized row, read `id`, `user_id`, `image_url` and `updated_at` into administrative process memory. Call `planInlineMedia(row, projectUrl)` for validation and the deterministic private destination. Call `migrateInlineMedia(admin, row, projectUrl)` only for that reviewed row. The function uploads without overwriting an existing object, verifies stored length/SHA256, then updates only `image_url` conditional on ID, owner and timestamp. A conflict is not success; keep the competing edit and investigate before retrying.

Retain an access-controlled receipt outside the repository containing the row/owner IDs, original timestamp and returned migration receipt, including the post-update timestamp. Receipts contain private object references and must not be committed. Original image bytes remain in the private content-addressed object; the receipt contains no image body or expiring bearer token. Do not delete these objects during the rollback window.

Verify owner download and byte digest after each canary, foreign/anonymous denial, metadata preservation and native display. Recheck pending sync behavior and payload reduction. Expand only after the canary's applicable checks succeed; any failed check stops further rows. No bulk migration or server policy change is part of the fixture scripts.

Rollback uses `restoreInlineMedia(admin, rowSnapshot, receipt, projectUrl)`. Use the migration's post-update row version when reversing an unedited canary. It verifies receipt ownership/origin/path, downloads and validates the backup, and conditionally restores only the image field. A stale timestamp or changed image reference must return conflict rather than overwrite a competing edit. Review any later edit before choosing a new snapshot; never blindly retry conflicts. Image bytes are restored exactly; base64 URI formatting is canonicalized. Other fields are retained. The private backup is deliberately kept until verification and the retention policy permit its removal.

Verify restored bytes, metadata and owner access, and account for all created objects. Run aggregate cleanup checks for fixture accounts/rows/objects after testing. Do not mistake fixture deletion for permission to remove customer backups.

Evidence lives in `docs/reports/readiness-evidence/production-inline-media-migration.json`, `production-inline-media-sync-compatibility.json`, and `production-inline-media-rollback.json`. These prove their stated fixture scopes, not a completed customer rollout, installed-app compatibility or 10,000-DAU capacity.
