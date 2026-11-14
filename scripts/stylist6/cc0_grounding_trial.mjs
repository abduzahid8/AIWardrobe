// Licensed internal calibration. Exactly one batch per pinned plan and arm.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import {
  createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION, NVIDIA_VISION_MODEL, verifiedVisualImage,
  VISUAL_TIMEOUT_MS, VISUAL_PROVIDER_TIMEOUT_MS, VISUAL_CONCURRENCY,
} from '../../backend/stylist6/visual_critic_hosted.mjs';

const { values } = parseArgs({ options: {
  plan: { type: 'string' }, arm: { type: 'string' }, sha256: { type: 'string' }, 'free-tier': { type: 'boolean' },
} });
if (!values.plan || !['control', 'grounded'].includes(values.arm)) throw new Error('expected plan and arm');
const file = fs.realpathSync(values.plan);
const directory = path.dirname(file);
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const raw = fs.readFileSync(file);
const planHash = digest(raw);
const plan = JSON.parse(raw);
const criticHash = digest(fs.readFileSync(new URL('../../backend/stylist6/visual_critic_hosted.mjs', import.meta.url)));
if (plan.model !== NVIDIA_VISION_MODEL || plan.criticSha256 !== criticHash ||
    !Array.isArray(plan.liveCases) || plan.liveCases.length < 1 || plan.liveCases.length > 4 || new Set(plan.liveCases).size !== plan.liveCases.length || plan.rights !== 'met-cc0') {
  throw new Error('invalid_calibration_plan');
}
const load = (record, relative) => {
  if (record.file !== relative) throw new Error('unexpected_calibration_asset');
  const target = path.join(directory, relative);
  if (fs.realpathSync(target) !== target) throw new Error('unexpected_calibration_media_path');
  const image = { bytes: fs.readFileSync(target), sha256: record.sha256 };
  verifiedVisualImage(image);
  return image;
};
const images = new Map(Object.entries(plan.assets).map(([id, record]) => {
  if (!/^\d+$/.test(id) || String(record.objectID) !== id || record.isPublicDomain !== true ||
      record.license !== 'CC0-1.0' || new URL(record.objectURL).hostname !== 'www.metmuseum.org' ||
      new URL(record.primaryImageSmall).hostname !== 'images.metmuseum.org') throw new Error('asset_rights_unverified');
  return [id, load(record, 'photos/' + id + '.jpg')];
}));
const reference = load(plan.reference, 'reference.jpg');
const slotSha256 = Object.fromEntries(plan.reference.sources.map((id) => [plan.assets[id].role, images.get(String(id)).sha256]));
const candidates = plan.cases.map(({ id, assets }) => {
  const products = assets.map((assetId) => {
    const key = String(assetId), asset = plan.assets[key];
    if (!asset) throw new Error('unlicensed_calibration_input');
    return { id: key, slot: asset.role, imageUrl: asset.primaryImageSmall };
  });
  const roles = new Set(products.map(({ slot }) => slot));
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(id) || ![3, 4].includes(products.length) ||
      roles.size !== products.length || !['top', 'bottom', 'shoes'].every((role) => roles.has(role))) {
    throw new Error('invalid_calibration_candidate');
  }
  return { id, products, referenceId: plan.reference.id };
});
if (new Set(candidates.map(({ id }) => id)).size !== candidates.length) throw new Error('duplicate_calibration_candidate');
const batch = plan.liveCases.map((id) => {
  const candidate = candidates.find((row) => row.id === id);
  if (!candidate) throw new Error('unknown_calibration_candidate');
  return candidate;
});
const config = {
  arm: values.arm, planHash, model: NVIDIA_VISION_MODEL, criticVersion: HOSTED_VISUAL_CRITIC_VERSION,
  maxCalls: batch.length, concurrency: VISUAL_CONCURRENCY, deadlineMs: VISUAL_TIMEOUT_MS, providerDeadlineMs: VISUAL_PROVIDER_TIMEOUT_MS,
};
if (!values['free-tier']) {
  console.log(JSON.stringify({ ...config, providerCalls: 0, casesVerified: candidates.length }, null, 2));
  process.exit(0);
}
const apiKey = process.env.NVIDIA_API_KEY;
if (!apiKey?.startsWith('nvapi-') || values.sha256 !== planHash) throw new Error('pinned_trial_credential_required');
// Stop cross-arm execution on account or rate-limit errors.
const prefix = path.join(directory, 'grounding-' + planHash);
const stop = path.join(directory, '.grounding-trial-stop.json');
if (fs.existsSync(stop) || fs.existsSync(prefix + '-stop.json')) throw new Error('trial_admission_stopped');
const active = path.join(directory, '.grounding-trial-active');
fs.writeFileSync(active, JSON.stringify({ arm: values.arm }), { flag: 'wx', mode: 0o600, flush: true });
try {
  const claim = prefix + '-' + values.arm + '.claim.json';
  fs.writeFileSync(claim, JSON.stringify(config), { flag: 'wx', mode: 0o600, flush: true });
  if (values.arm === 'grounded') reference.components = { referenceSha256: reference.sha256, slotSha256 };
  const events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: (id) => id === plan.reference.id ? reference : null,
    resolveProduct: ({ id, imageUrl }) => plan.assets[id]?.primaryImageSmall === imageUrl ? images.get(id) : null,
    fetchImpl: async (url, options) => {
      if (url !== 'https://integrate.api.nvidia.com/v1/chat/completions' || options.method !== 'POST' ||
          options.redirect !== 'error' || calls >= config.maxCalls) throw new Error('trial_dispatch_blocked');
      calls++;
      const response = await fetch(url, options);
      if ([401, 402, 403, 429].includes(response.status) && !fs.existsSync(stop)) {
        fs.writeFileSync(stop, JSON.stringify({ httpStatus: response.status }), { flag: 'wx', mode: 0o600 });
      }
      return response;
    },
    onEvent: (event) => events.push(event),
  });
  const started = performance.now();
  const verdicts = await critic(batch, plan.context);
  const receipt = { ...config, calls, wallMs: performance.now() - started, events, verdicts };
  fs.writeFileSync(prefix + '-' + values.arm + '.receipt.json', JSON.stringify(receipt, null, 2) + '\n',
    { flag: 'wx', mode: 0o600, flush: true });
  console.log(JSON.stringify({ ...config, calls, wallMs: receipt.wallMs, valid: verdicts.length, events }, null, 2));
} finally {
  fs.unlinkSync(active);
}
