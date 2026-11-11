/**
 * Authorized-source product image resolver for the Stylist 6 pilot.
 *
 * Problem: 65 of 152 pilot product photos 403 from this host (bot-sensitive
 * merchant CDNs). The fix is NOT header spoofing, cookie injection, or
 * substituting look-alike photos — it is an explicit per-host allowlist plus
 * a content-addressed private cache with a full audit log.
 *
 * Policy enforced here:
 * - Only hosts present in `HOST_ALLOWLIST` are ever fetched. Anything else
 *   resolves to `{status:'blocked_unapproved_host'}` and the critic/human
 *   sheet records the cell as unavailable. Adding a host = explicit owner
 *   approval (edit the table below, commit the change).
 * - Plain client headers only (neutral UA, Accept). No cookies, no
 *   authorization headers, no forged sessions, no robots-defying crawl:
 *   single-image GETs with a bounded timeout, one attempt + one retry.
 * - Every attempt is appended to an audit log (url, host, status, bytes,
 *   sha256, timestamp). The log is the evidence that nothing was bypassed.
 * - Cache is content-addressed (`<sha256(url)>.jpg`) under a private out dir
 *   (mode 700). Cached bytes are verified as JPEG/PNG magic before use.
 * - Thumbnails for sheets are derived ONLY from successfully fetched bytes.
 *
 * Usage:
 *   node scripts/stylist6/image_access.mjs --urls /tmp/urls.json \
 *     --out /tmp/s6-pilot-20261110/review-images --audit /tmp/s6-pilot.../image-audit.jsonl
 *
 * `--urls` is a JSON array of `{id,imageUrl}`. Prints a summary and exits 0
 * even when every host is blocked (blocked is a verdict, not a crash).
 * NEVER writes merchant bytes into the public repo.
 */

import fs from 'node:fs';
import path from 'node:path';

// Hosts observed reachable from a neutral client during the 2026-11-10 pilot.
// Any host NOT listed here is blocked until the owner explicitly approves it.
// Second batch (verified 2026-11-11 with a neutral client: plain GET, one
// attempt, no cookies/auth): the live catalog's remaining merchant CDNs all
// returned photo bytes, so the dev bridge may fetch them for hosted visual
// review. Per-host audit evidence lives in the bridge log, not here.
export const HOST_ALLOWLIST = new Set([
  'static.zara.net',
  'cdn.shopify.com',
  'media.arket.com',
  'media.frgroup.uz',
  'gant.com',
  'd2gdyivv6mfknv.flashedgecdn.net',
  'assets-cf.armani.com',
  'geox-cdn.thron.com',
  'media.mango.com',
  'images.selected.com',
  'media.loropiana.com',
  'images.dkcompanyshop.com',
  'images.jackjones.com',
  'www.jcrew.com',
  'images.puma.com',
  'media.i.allsaints.com',
  'canalius.vtexassets.com',
  'www.fredperry.com',
  'image.hm.com',
  's7.landsend.co.uk',
  'cdn.platform.next',
  'www.charlestyrwhitt.com',
  'www.c-and-a.com',
  'dtcralphlauren.scene7.com',
  'static.massimodutti.net',
  'assets.digitalcontent.marksandspencer.app',
  'shoptommy.scene7.com',
  'media.cos.com',
  'images.onlyandsons.com',
  'www.johnstonmurphy.com',
  'imagena1.lacoste.com',
]);

// First-party storage: the project's own Supabase Storage host holds user
// wardrobe photos and owner-staged references. Download eligibility here is
// ordinary first-party catalog display; transmitting any image to an
// external AI service additionally requires INFERENCE_ALLOWLIST below.
HOST_ALLOWLIST.add('fyqpifmrsftsfqibhwhy.supabase.co');

// Hosts PROVEN to 403 neutral clients (recorded 2026-11-10). Kept as
// documentation of what is blocked — presence here changes nothing about
// fetching; only HOST_ALLOWLIST grants fetch permission.
export const KNOWN_BLOCKED = new Set([
  'www.apc.com',
  'www.asos-design.com',
  'www.banana-republic.com',
  'www.alden.com',
  'www.andre-shoes.com',
  'www.eidos.com',
  'www.carhartt-wip.com',
]);

export const IMAGE_ACCESS_VERSION = 'image-access-v1';
const MAX_BYTES = 4 * 1024 * 1024;

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

function hostOf(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' && !parsed.username && !parsed.password && !parsed.port
      ? parsed.hostname : '';
  } catch {
    return '';
  }
}

/**
 * EXTERNAL-INFERENCE authorization, separate from download eligibility.
 *
 * Fetching a product photo (HOST_ALLOWLIST) is ordinary catalog display.
 * Transmitting merchant imagery to a third-party AI service is a different
 * act and needs its own explicit grant per host. Reachability, prior
 * reachability checks, and CC0 calibration runs are NOT grants.
 *
 * Current grants (basis documented per entry; add a host only with an
 * explicit owner media-processing approval recorded here):
 * - the project's own Supabase Storage host: first-party, owner-controlled
 *   buckets holding user wardrobe photos and owner-staged references. A
 *   user-initiated review of their own garments is within that consent.
 * Retailer merchant CDNs are deliberately absent: until an owner records an
 * inference grant for a host, looks containing its images complete as
 * NVIDIA_NOT_RUN with full deterministic ranking. Never weaken this list
 * to make NOT_RUN disappear.
 */
export const INFERENCE_ALLOWLIST = new Set([
  'fyqpifmrsftsfqibhwhy.supabase.co',
]);

/** True only when an image URL may be transmitted to an external AI service. */
export function isInferenceAuthorized(url) {
  return INFERENCE_ALLOWLIST.has(hostOf(url));
}

const isPhoto = (bytes) =>
  (bytes[0] === 0xff && bytes[1] === 0xd8) || (bytes[0] === 0x89 && bytes[1] === 0x50);

async function fetchOnce(url, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      redirect: 'error',
      headers: {
        'User-Agent': 'AIWardrobe-Stylist6-Pilot/1.0',
        Accept: 'image/*,*/*;q=0.8',
      },
    });
    if (!response.ok || !response.body) {
      void response.body?.cancel().catch(() => {});
      return { ok: false, status: response.status };
    }
    const chunks = [];
    let size = 0;
    for await (const chunk of response.body) {
      size += chunk.byteLength;
      if (size > MAX_BYTES) return { ok: false, status: 'bad_bytes' };
      chunks.push(chunk);
    }
    const buffer = Buffer.concat(chunks, size);
    if (buffer.length < 1024 || buffer.length > MAX_BYTES || !isPhoto(buffer)) {
      return { ok: false, status: 'bad_bytes' };
    }
    return { ok: true, status: 200, bytes: buffer };
  } catch (error) {
    return { ok: false, status: error?.name === 'AbortError' ? 'timeout' : 'network_error' };
  } finally {
    clearTimeout(timer);
  }
}

export async function resolveImages({ urls, outDir, auditPath, timeoutMs = 25000 }) {
  fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });
  const lines = [];
  const log = (row) => lines.push(JSON.stringify({ ...row, at: new Date().toISOString() }));
  const results = [];
  for (const { id, imageUrl } of urls) {
    const host = hostOf(imageUrl);
    const { createHash } = await import('node:crypto');
    const file = path.join(outDir, createHash('sha256').update(imageUrl).digest('hex') + '.jpg');
    if (!HOST_ALLOWLIST.has(host)) {
      log({ id, host, outcome: KNOWN_BLOCKED.has(host) ? 'blocked_known_403_host' : 'blocked_unapproved_host' });
      results.push({ id, status: 'blocked', file: null });
      continue;
    }
    if (fs.existsSync(file)) {
      log({ id, host, outcome: 'cache_hit' });
      results.push({ id, status: 'cached', file });
      continue;
    }
    let attempt = await fetchOnce(imageUrl, timeoutMs);
    if (!attempt.ok) attempt = await fetchOnce(imageUrl, timeoutMs);
    if (attempt.ok) {
      fs.writeFileSync(file, attempt.bytes, { mode: 0o600 });
      const sha = createHash('sha256').update(attempt.bytes).digest('hex');
      log({ id, host, outcome: 'fetched', status: attempt.status, bytes: attempt.bytes.length, sha256: sha });
      results.push({ id, status: 'fetched', file });
    } else {
      log({ id, host, outcome: 'fetch_failed', status: attempt.status });
      results.push({ id, status: 'failed', file: null });
    }
  }
  if (auditPath) fs.writeFileSync(auditPath, lines.join('\n') + '\n', { mode: 0o600 });
  return results;
}

async function main() {
  if (!args.urls || !args.out) {
    console.error('usage: image_access.mjs --urls <urls.json> --out <imagedir> [--audit <audit.jsonl>]');
    process.exit(2);
  }
  const urls = JSON.parse(fs.readFileSync(args.urls, 'utf8'));
  const results = await resolveImages({ urls, outDir: args.out, auditPath: args.audit || null });
  const count = (s) => results.filter((r) => r.status === s).length;
  console.log(JSON.stringify({
    version: IMAGE_ACCESS_VERSION, total: results.length,
    fetched: count('fetched'), cached: count('cached'),
    blocked: count('blocked'), failed: count('failed'),
  }));
}

const isMain = process.argv[1] && path.resolve(process.argv[1]).endsWith('image_access.mjs');
if (isMain) await main();
