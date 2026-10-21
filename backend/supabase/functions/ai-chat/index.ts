import { enforceRateLimit } from '../_shared/rateLimit.ts'
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

/**
 * ai-chat — conversational stylist, Supabase-native replacement for
 * Express POST /api/ai-chat (backend/api/routes/ai/chat.js).
 *
 * Contract (kept identical so mobile clients switch by URL only):
 *   Request:  { query: string (1..1000), conversationHistory?: [{role, content}], stylePreference?: string }
 *   Success:  { text: string, provider: 'openai' | 'gemini' | 'nvidia' | 'huggingface' }
 *   Failure:  400 validation / 401 auth / 503 all-providers-down
 *             (the mobile client falls back to the offline Ava persona).
 *
 * Auth: Supabase JWT required (mirrors authenticateToken).
 * Tier routing mirrors generate-outfits: paid (lite/premium/vip,
 * active/trial) tries Gemini first; free/anonymous tries NVIDIA first,
 * then Gemini on quota failures. OpenAI is attempted first wherever a
 * key exists (cheapest fast path when configured).
 */

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const STYLIST_SYSTEM = `You are Ava, a world-class personal stylist. Warm, confident, concise: 1-3 short sentences, no fluff. Always actionable — name real pieces, colors, and fit. Wardrobe-first: reference owned items before suggesting shop buys. One question at a time.

UNIVERSAL STYLE PRINCIPLES: Fit is king (medium-tapered, shoulders align). Loafers (penny/bit/tassel), oxfords, derbies for smart and classic; driving moccasins sit a tier below loafers — Riviera-casual, never with tailoring. Never mix formality tiers (no sneakers with suits). Black pairs with white only. Max three colors per outfit. Socks match trousers, not shoes. Graphic prints, hoodies, cargos, and chunky sneakers never belong in classic looks.`

function buildSystemPrompt(stylePreference?: string | null): string {
    const style = String(stylePreference || 'casual').toLowerCase().replace(/[\s-]+/g, '_')
    const hints: Record<string, string> = {
        old_money: 'Style direction: Old Money quiet luxury — tonal, textured, timeless. Navy, cream, camel, beige, burgundy. Blazer/cardigan + knit polo/oxford + pleated trousers + loafers.',
        semi_classic: 'Style direction: refined everyday — Massimo Dutti register. Cardigan or unstructured blazer + knit polo + chinos + loafers or clean sneakers.',
        minimalist: 'Style direction: minimalist — tight palette (black/white/grey/beige/navy), monochrome, texture over color. No prints or logos.',
        business_casual: 'Style direction: modern professional — textured blazer + knit polo + tailored trousers + loafers/derbies. Never hoodies, graphics, or sneakers at work.',
        classic: 'Style direction: classic menswear — Savile-Row rooted tailoring, high-rise trousers, oxfords/derbies/loafers. Never streetwear with tailoring.',
        casual: 'Style direction: smart casual — knit tee/polo + chinos or dark jeans + minimal sneakers or loafers.',
    }
    return `${STYLIST_SYSTEM}\n\n${hints[style] || hints['casual']}`
}

type ChatMsg = { role: 'system' | 'user' | 'assistant'; content: string }

async function callOpenAI(key: string, messages: ChatMsg[], maxTokens: number): Promise<string | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 25_000)
    try {
        const r = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
            body: JSON.stringify({ model: 'gpt-4o-mini', messages, max_tokens: maxTokens, temperature: 0.8 }),
            signal: controller.signal,
        })
        clearTimeout(timer)
        if (!r.ok) return null
        const d = await r.json()
        const text = d.choices?.[0]?.message?.content?.trim()
        return text || null
    } catch {
        clearTimeout(timer)
        return null
    }
}

async function callGemini(key: string, model: string, sys: string, messages: ChatMsg[], maxTokens: number): Promise<string | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 25_000)
    try {
        const contents = messages
            .filter((m) => m.role !== 'system')
            .map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] }))
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                system_instruction: { parts: [{ text: sys }] },
                contents,
                generationConfig: { temperature: 0.8, maxOutputTokens: maxTokens },
            }),
            signal: controller.signal,
        })
        clearTimeout(timer)
        if (!r.ok) return null
        const d = await r.json()
        const text = d.candidates?.[0]?.content?.parts?.map((p: any) => p.text || '').join('').trim()
        return text || null
    } catch {
        clearTimeout(timer)
        return null
    }
}

async function callNvidia(key: string, model: string, messages: ChatMsg[], maxTokens: number): Promise<string | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 20_000)
    try {
        const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
            body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.8 }),
            signal: controller.signal,
        })
        clearTimeout(timer)
        if (!r.ok) return null
        const d = await r.json()
        const text = d.choices?.[0]?.message?.content?.trim()
        return text || null
    } catch {
        clearTimeout(timer)
        return null
    }
}

async function callHuggingFace(key: string, prompt: string, maxTokens: number): Promise<string | null> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 25_000)
    try {
        const r = await fetch('https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
            body: JSON.stringify({ inputs: prompt, parameters: { max_new_tokens: maxTokens, temperature: 0.8, return_full_text: false } }),
            signal: controller.signal,
        })
        clearTimeout(timer)
        if (!r.ok) return null
        const d = await r.json()
        const text = (Array.isArray(d) ? d[0]?.generated_text : d.generated_text || '')?.trim()
        return text || null
    } catch {
        clearTimeout(timer)
        return null
    }
}

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    const rateLimitResponse = await enforceRateLimit(req, { feature: 'ai-chat', limit: 20, headers: corsHeaders })
    if (rateLimitResponse) return rateLimitResponse

    const json = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })

    const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
    const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

    // ── Auth (mirrors Express authenticateToken) ──
    const authHeader = req.headers.get('Authorization') || ''
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : ''
    if (!token) return json({ error: 'Authentication required', code: 'NO_TOKEN' }, 401)
    const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        global: { headers: { Authorization: authHeader } },
    })
    const { data: { user }, error: authErr } = await userClient.auth.getUser()
    if (authErr || !user) return json({ error: 'Invalid or expired token', code: 'INVALID_TOKEN' }, 401)

    // ── Validate (mirrors validateAIChat) ──
    let body: any = {}
    try {
        body = await req.json()
    } catch {
        return json({ error: 'Validation failed', details: 'Invalid JSON body' }, 400)
    }
    const query = typeof body.query === 'string' ? body.query.trim() : ''
    if (!query) return json({ error: 'Validation failed', details: 'Query is required' }, 400)
    if (query.length > 1000) return json({ error: 'Validation failed', details: 'Query must be less than 1000 characters' }, 400)
    const history: Array<{ role: string; content: string }> = Array.isArray(body.conversationHistory)
        ? body.conversationHistory
            .filter((m: any) => m && typeof m.content === 'string')
            .slice(-10)
            .map((m: any) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content.slice(0, 1000) }))
        : []
    const stylePreference = typeof body.stylePreference === 'string' ? body.stylePreference.slice(0, 60) : undefined

    // ── Keys ──
    const openaiKey = Deno.env.get('OPENAI_API_KEY') || ''
    let nvidiaKey = Deno.env.get('NVIDIA_API_KEY') || ''
    const nvidiaModel = (Deno.env.get('NVIDIA_TEXT_MODEL') || Deno.env.get('NVIDIA_MODEL') || '').trim()
        || 'meta/llama-3.2-11b-vision-instruct'
    let geminiKey = Deno.env.get('GEMINI_API_KEY') || ''
    const geminiModelPrimary = (Deno.env.get('GEMINI_MODEL') || '').trim() || 'gemini-3.5-flash-lite'
    let hfKey = Deno.env.get('HF_TOKEN') || ''
    if (SUPABASE_SERVICE_ROLE_KEY && (!nvidiaKey || !geminiKey || !hfKey)) {
        try {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: cfg } = await svc.from('app_config').select('key, value')
                .in('key', ['nvidia_token', 'gemini_token', 'gemini_model', 'hf_token'])
            for (const row of cfg || []) {
                if (row.key === 'nvidia_token' && !nvidiaKey) nvidiaKey = row.value
                if (row.key === 'gemini_token' && !geminiKey) geminiKey = row.value
                if (row.key === 'hf_token' && !hfKey) hfKey = row.value
            }
        } catch {
            // fall through to whatever keys we have
        }
    }
    const geminiModels = [...new Set([geminiModelPrimary, 'gemini-2.0-flash'].filter(Boolean))]

    // ── Tier (server-authoritative; fail-open to free) ──
    let paidTier = false
    if (SUPABASE_SERVICE_ROLE_KEY) {
        try {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: subs } = await svc.from('subscriptions').select('tier, status')
                .eq('user_id', user.id).order('created_at', { ascending: false }).limit(5)
            paidTier = (subs || []).some((s: any) =>
                String(s.tier || '').toLowerCase() !== 'free'
                && ['active', 'trial'].includes(String(s.status || '').toLowerCase()),
            )
        } catch {
            // fail open
        }
    }

    // ── Provider chain ──
    const sys = buildSystemPrompt(stylePreference)
    const messages: ChatMsg[] = [
        { role: 'system', content: sys },
        ...history.map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
        { role: 'user', content: query },
    ]
    const tryGemini = async (): Promise<{ text: string; provider: string } | null> => {
        if (!geminiKey) return null
        for (const gm of geminiModels) {
            const text = await callGemini(geminiKey, gm, sys, messages, 500)
            if (text) return { text, provider: 'gemini' }
        }
        return null
    }

    // NVIDIA is the configured provider for every subscription tier.
    const text = nvidiaKey ? await callNvidia(nvidiaKey, nvidiaModel, messages, 500) : null
    const result = text ? { text, provider: 'nvidia' } : null

    if (!result) {
        return json({ error: 'AI model is busy, try again later.' }, 503)
    }
    return json({ text: result.text, provider: result.provider })
})
