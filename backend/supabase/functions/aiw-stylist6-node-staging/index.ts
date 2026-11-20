// Stylist 6 QA gateway: use the verified 75s Vercel Fluid Compute path
// (maxDuration: 300s) with 85s proxy timeout so authenticated iOS staging
// exercises the full Autonomous Visual Stylist (Act 1 Showroom + Act 2 Editor
// + Parallel Swaps) with zero Render sleep cold starts.
//
// Preserves Supabase JWT verification, dedicated staging HMAC, identical
// no-retry at-most-once quota semantics, and all server source/budget gates.
// Production generate-outfits (Stylist 5) is NOT changed.
//
// Rollback: switch import back to stylist6-render-staging.ts (or invoke
// aiw-stylist6-node-render-staging) and redeploy ONLY aiw-stylist6-node-staging.
import "../../../stylist-proxy/stylist6-staging.ts";
