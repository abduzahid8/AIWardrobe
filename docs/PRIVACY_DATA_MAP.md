# PRIVACY_DATA_MAP.md (Phase 7CZ, 7T)

Classification → handling. Code: `src/services/launch/privacy.ts`,
`src/services/growth/*` (sanitized-by-construction).

| Class | Examples | Logs | External AI | Export | Delete |
|---|---|---|---|---|---|
| public | catalog listings | ok | ok | n/a | n/a |
| account | display name, email | id only | no | yes | yes |
| wardrobe_private | items, outfits, plans | never contents | no (aggregates only w/ consent) | yes | yes |
| body_fit_private | measurements, fit feedback | never | **never raw** | yes | yes |
| purchase_private | records, watches | never | never | yes | yes |
| image_private | wardrobe/reference photos | never | never raw; on-device parse preferred | metadata only | yes + storage objects |
| operational | errors (sanitized), flags | sanitized | n/a | no | anonymize |
| analytics | events (allowlisted) | allowlisted | n/a | no | anonymize identifiers |

Notifications default: in-app cards; push only for opted-in categories.
External AI processing control defaults **off**; enabling it does not lift
the raw-private ban (only derived features cross the boundary, with consent).
