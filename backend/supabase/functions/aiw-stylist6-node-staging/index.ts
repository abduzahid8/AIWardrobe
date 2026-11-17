// Stylist 6 QA gateway: use the verified 72s Render compute path so
// authenticated iOS staging actually exercises AI SKU composition and the
// final independent full-photo fashion editor. The 50s Vercel path cannot
// enable those stages (cloud_handler gates them at >=70s).
//
// Both gateways retain Supabase JWT verification, dedicated staging HMAC,
// identical no-retry at-most-once quota semantics, and source/budget gates.
// Production generate-outfits (Stylist 5) is NOT changed.
//
// Rollback: restore the previous import of stylist6-staging.ts and redeploy
// ONLY aiw-stylist6-node-staging; the separate render staging stays intact.
import "../../../stylist-proxy/stylist6-render-staging.ts";
