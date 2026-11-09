/**
 * Rebundle the EXACT Supabase Stylist 5 implementation for the Node runtime.
 * The Deno handler, fashion rules and quota logic are NOT duplicated.
 *
 * Only three platform-specific HTTP imports are adapted at build time:
 *   Deno's serve() -> capture the request handler for Vercel
 *   Deno esm.sh supabase-js -> the same pinned Node SDK package
 *
 * A build failure is preferable to silently shipping a stale or partial core.
 */
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const serviceRoot = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(serviceRoot, '..', '..');
const edgeEntry = resolve(repoRoot, 'backend/supabase/functions/generate-outfits/index.ts');
const outFile = resolve(serviceRoot, 'dist/stylist-bundle.mjs');

const source = await readFile(edgeEntry, 'utf8');
if (!source.includes('GENERATION_VERSION') ||
    !source.includes('enforceRateLimit') ||
    !source.includes('createOutfitPlan(') ||
    !source.includes('serve(async(req:Request)=>')) {
  throw new Error('stylist_node_source_contract_changed');
}

const sha = createHash('sha256').update(source).digest('hex');
await mkdir(dirname(outFile), { recursive: true });
await build({
  entryPoints: [edgeEntry],
  outfile: outFile,
  bundle: true,
  platform: 'node',
  target: ['node22'],
  tsconfigRaw: { compilerOptions: { target: 'ES2022', useDefineForClassFields: true } },
  format: 'esm',
  packages: 'external',
  logLevel: 'info',
  legalComments: 'none',
  sourcemap: false,
  treeShaking: true,
  plugins: [{
    name: 'deno-stylist-adapter',
    setup(plugin) {
      plugin.onResolve(
        { filter: /^https:\/\/deno\.land\/std@0\.168\.0\/http\/server\.ts$/ },
        () => ({ path: 'aiwardrobe-serve-adapter', namespace: 'aiw-platform' })
      );
      plugin.onLoad(
        { filter: /^aiwardrobe-serve-adapter$/, namespace: 'aiw-platform' },
        () => ({ contents: 'export function serve(handler) { globalThis.__AIW_STYLIST_HANDLER__ = handler; }', loader: 'js' })
      );
      plugin.onResolve(
        { filter: /^https:\/\/esm\.sh\/@supabase\/supabase-js@2(?:\.45\.4)?$/ },
        () => ({ path: '@supabase/supabase-js', external: true })
      );
      plugin.onResolve(
        { filter: /^https:\/\// },
        args => { throw new Error('Unexpected remote import in Stylist: ' + args.path); }
      );
    }
  }],
});
console.info('stylist_node_bundle_ready', {
  sourceSha256: sha,
  entry: 'backend/supabase/functions/generate-outfits/index.ts',
  authAndQuotaSource: 'unchanged',
});
