/** Run the hosted handler against real services without changing hosted routing. */
import 'dotenv/config';
import ts from 'typescript';
import { readFile } from 'node:fs/promises';
import { createClient } from '@supabase/supabase-js';
const originalFetch = globalThis.fetch;
if (!process.argv[2]) throw new Error('Pass the saved hosted generate-outfits TypeScript entrypoint as the first argument');
const source = (await readFile(process.argv[2], 'utf8'))
  .replace(/^import .*\n/gm, '').replace(/\bexport\s+/g, '')
  .replace("geminiModelPrimary = 'gemini-3.5-flash'", "geminiModelPrimary = 'gemini-3.5-flash-lite'");
const helper = (await readFile('backend/supabase/functions/_shared/modelOutfitResponse.ts', 'utf8')).replace(/\bexport\s+/g, '');
const compiled = ts.transpileModule(helper + '\n' + source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
let handler;
const variables = {
  SUPABASE_URL: process.env.EXPO_PUBLIC_SUPABASE_URL,
  SUPABASE_ANON_KEY: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,
  SUPABASE_SERVICE_ROLE_KEY: process.env.SUPABASE_SERVICE_KEY,
  GEMINI_API_KEY: process.env.GEMINI_TEST_KEY,
};
if (!variables.GEMINI_API_KEY || !variables.SUPABASE_SERVICE_ROLE_KEY) throw new Error('Test credentials required');
new Function('serve', 'createClient', 'Deno', compiled)(fn => { handler = fn; }, createClient, { env: { get: name => variables[name] } });
if (!handler) throw new Error('Hosted handler not captured');
globalThis.fetch = (input, init) => {
  const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
  if (url.origin === new URL(variables.SUPABASE_URL).origin && url.pathname === '/functions/v1/generate-outfits') {
    return handler(new Request(input, init));
  }
  return originalFetch(input, init);
};
try { await import('./production-ai-smoke.mjs'); }
finally { globalThis.fetch = originalFetch; }
