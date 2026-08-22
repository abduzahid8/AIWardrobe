// =============================================================================
// Virtual Try-On — Deterministic mannequin renderer
// =============================================================================
// The mannequin never changes pose, so garment geometry is rendered
// deterministically with mannequin-specific anchor boxes and layering rules.
// This produces stable outfit combinations without regenerating the mannequin.
//
// Pipeline per garment step:
//   1) Remove garment background and crop to garment silhouette.
//   2) Place it into a category-specific anchor box on the fixed mannequin.
//   3) Apply deterministic category shaping, shadows, and occlusion rules.
// =============================================================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })

  const respond = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })

  return respond({
    success: false,
    error: 'This edge function no longer performs deterministic rendering. Use the Node API POST /api/tryon/render which runs mannequin-locked FLUX.1-Kontext-dev via poseAnchoredRender (single-call per outfit).',
    migration: '/api/tryon/render',
  }, 410)
})
