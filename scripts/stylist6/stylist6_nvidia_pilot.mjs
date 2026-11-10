/**
 * Isolated fixed-pool V5/V6 pilot. Default mode plans with zero network I/O.
 * The organizer bundle is private and stays outside the repository.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { trialImageDecision } from '../../backend/stylist6/retail_evidence.mjs';
import { adaptCatalog } from '../../backend/stylist6/catalog_adapter.mjs';
import { evaluateVisualShortlist, STYLIST6_VERSION } from '../../backend/stylist6/prototype.mjs';
import {
  createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION, MAX_VISUAL_IMAGE_BYTES,
  MAX_VISUAL_CANDIDATES, NVIDIA_VISION_MODEL, VISUAL_TIMEOUT_MS, VISUAL_PROVIDER_TIMEOUT_MS, VISUAL_CONCURRENCY, verifiedVisualImage,
} from '../../backend/stylist6/visual_critic_hosted.mjs';

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const readJson = (bytes) => JSON.parse(bytes.toString('utf8'));
const asJson = (value) => JSON.stringify(value, null, 2) + '\n';
const SLOTS = ['top', 'outerwear', 'bottom', 'shoes'];

function readBounded(file, limit) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
  try {
    const info = fs.fstatSync(fd);
    if (!info.isFile() || info.size > limit) throw new Error('invalid_pilot_file');
    const bytes = Buffer.alloc(info.size);
    if (fs.readSync(fd, bytes, 0, bytes.length, 0) !== info.size || fs.fstatSync(fd).size !== info.size) {
      throw new Error('pilot_file_changed');
    }
    return bytes;
  } finally {
    fs.closeSync(fd);
  }
}

export function loadPilotCase(directory, scenarioId) {
  const readManifest = (name) => readBounded(path.join(directory, name), 1024 * 1024);
  const queueBytes = readManifest('paired_critic_queue.json');
  const inventoryBytes = readManifest('inventory.json');
  const scenariosBytes = readManifest('scenarios.json');
  const queue = readJson(queueBytes);
  const scenario = readJson(scenariosBytes).find(({ id }) => id === scenarioId);
  const entry = queue.cases.find(({ scenario }) => scenario.id === scenarioId);
  if (!scenario || !entry || JSON.stringify(scenario) !== JSON.stringify(entry.scenario)
      || queue.inventorySha256 !== sha(inventoryBytes) || queue.scenariosSha256 !== sha(scenariosBytes)
      || !/^rt_\d{3}$/.test(scenario.reference)
      || (scenario.budget && (!Number.isFinite(scenario.budget.amount) || scenario.budget.amount <= 0
          || !/^[A-Z]{3}$/.test(scenario.budget.currency)))) {
    throw new Error('pilot_manifest_mismatch');
  }
  if (entry.candidates.length !== MAX_VISUAL_CANDIDATES
      || entry.candidates.filter(({ origin }) => origin === 'v5').length > 1) {
    throw new Error('pilot_candidate_count_mismatch');
  }
  const rows = readJson(inventoryBytes);
  const products = new Map(rows.map((asset) => [asset.row.id, asset]));
  const { items } = adaptCatalog(rows.map(({ row }) => row));
  const catalog = new Map(items.map((item) => [item.id, item]));
  if (products.size !== rows.length || catalog.size !== items.length) throw new Error('duplicate_catalog_sku');

  const referenceHashes = readJson(readManifest('references/hashes.json'));
  let rightsBytes;
  try { rightsBytes = readManifest('image-rights.json'); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const rights = rightsBytes ? readJson(rightsBytes) : null;
  const reference = {
    bytes: readBounded(path.join(directory, 'references', scenario.reference + '.jpg'), MAX_VISUAL_IMAGE_BYTES),
    sha256: referenceHashes[scenario.reference],
  };
  verifiedVisualImage(reference);
  const photos = new Map();
  const identities = new Set(), candidateIds = new Set();
  const candidates = entry.candidates.map((candidate) => {
    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(candidate.id) || candidateIds.has(candidate.id)
        || !Array.isArray(candidate.items) || candidate.items.length !== SLOTS.length
        || new Set(candidate.items).size !== SLOTS.length) throw new Error('pilot_candidate_invalid');
    candidateIds.add(candidate.id);
    const items = candidate.items.map((id) => {
      const asset = products.get(id), item = catalog.get(id);
      if (!item || asset?.row.is_active !== true || item.imageUrl !== asset.row.image_url
          || (scenario.budget && item.currency !== scenario.budget.currency)) {
        throw new Error('pilot_catalog_ineligible');
      }
      if (!photos.has(id)) {
        if (!/^photos\/[0-9a-f]{64}\.jpg$/.test(asset.photo)) throw new Error('pilot_photo_path_invalid');
        const photo = { bytes: readBounded(path.join(directory, asset.photo), MAX_VISUAL_IMAGE_BYTES), sha256: asset.photoSha256 };
        verifiedVisualImage(photo);
        photos.set(id, photo);
      }
      return item;
    });
    const identity = JSON.stringify([...candidate.items].sort());
    if (identities.has(identity) || !SLOTS.every((slot) => items.filter((i) => i.macroCategory === slot).length === 1)
        || (scenario.budget && items.reduce((total, i) => total + i.price, 0) > scenario.budget.amount)) {
      throw new Error('pilot_outfit_ineligible');
    }
    identities.add(identity);
    return {
      id: candidate.id, origin: candidate.origin, briefId: scenario.reference,
      brief: { id: scenario.reference, name: scenario.style, referenceId: scenario.reference },
      items: SLOTS.map((slot) => items.find((item) => item.macroCategory === slot)),
    };
  });
  const request = { style: scenario.style, occasion: scenario.occasion, brief: scenario.brief, referenceId: scenario.reference };
  const plan = {
    scenarioId, request, budget: scenario.budget, model: NVIDIA_VISION_MODEL,
    criticVersion: HOSTED_VISUAL_CRITIC_VERSION, selectorVersion: STYLIST6_VERSION,
    timeoutMs: VISUAL_TIMEOUT_MS, providerTimeoutMs: VISUAL_PROVIDER_TIMEOUT_MS,
    maxProviderCalls: candidates.length, maxConcurrent: VISUAL_CONCURRENCY, retries: 0,
    implementationSha256: sha(Buffer.concat([
      import.meta.url,
      new URL('../../backend/stylist6/prototype.mjs', import.meta.url),
      new URL('../../backend/stylist6/visual_critic_hosted.mjs', import.meta.url),
      new URL('../../backend/stylist6/catalog_adapter.mjs', import.meta.url),
      new URL('../../backend/stylist6/retail_evidence.mjs', import.meta.url),
    ].map((file) => fs.readFileSync(new URL(file))))),
    rightsSha256: rightsBytes ? sha(rightsBytes) : null,
    queueSha256: sha(queueBytes), inventorySha256: sha(inventoryBytes), referenceSha256: reference.sha256,
    candidates: candidates.map(({ id, items }) => ({
      id, products: items.map((item) => ({
        id: item.id, slot: item.macroCategory, price: item.price, currency: item.currency,
        photoSha256: photos.get(item.id).sha256,
      })),
    })),
  };
  const rightsAssets = [
    { kind: 'reference', id: scenario.reference, sha256: reference.sha256 },
    ...[...photos].map(([id, photo]) => ({ kind: 'product', id, sha256: photo.sha256, sourceUrl: catalog.get(id).imageUrl })),
  ];
  return { plan, approvalSha256: sha(asJson(plan)), candidates, reference, photos, rights, rightsAssets };
}

export async function judgePilotCase(fixture, { apiKey, fetchImpl }) {
  const { candidates, plan, reference, photos } = fixture;
  if (!fetchImpl && (!fixture.rightsAssets?.length ||
      fixture.rightsAssets.some((asset) => trialImageDecision(fixture.rights, asset) !== 'cleared'))) {
    throw new Error('pilot_image_rights_denied');
  }
  const events = [];
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, fetchImpl, onEvent: (event) => events.push(event),
    resolveReference: (id) => id === plan.request.referenceId ? reference : null,
    resolveProduct: ({ id, imageUrl }) => candidates.some((c) =>
      c.items.some((item) => item.id === id && item.imageUrl === imageUrl)) ? photos.get(id) : null,
  });
  let verdicts = [];
  const retrieval = { generationVersion: STYLIST6_VERSION, shortlist: candidates, diagnostics: {} };
  const started = performance.now();
  const deck = await evaluateVisualShortlist(retrieval, plan.request, async (payload, request) => {
    verdicts = await critic(payload, request);
    // The V5 control is scored for fixed-pool calibration. Keep it separate
    // from the actual V6 arm, whose candidates come from V6 retrieval.
    return verdicts;
  }, { target: 1 });
  const visualOnly = candidates.filter(({ origin }) => origin !== 'v5');
  const v6Deck = await evaluateVisualShortlist({ ...retrieval, shortlist: visualOnly },
    plan.request, async () => verdicts.filter(({ candidateId }) => visualOnly.some(({ id }) => id === candidateId)), { target: 3 });
  return {
    providerCalls: events.filter(({ dispatched }) => dispatched).length,
    wallMs: performance.now() - started, providerCostUsd: null,
    v5Control: candidates.find(({ origin }) => origin === 'v5')?.id ?? null,
    v6RetrievalPick: visualOnly[0].id,
    fixedPoolPick: deck.outfits[0]?.id ?? null,
    v6VisualPick: v6Deck.outfits[0]?.id ?? null,
    v6VisualDeck: v6Deck.outfits.map(({ id, items }) => ({ id, items: items.map((item) => item.id) })),
    fixedPoolDiagnostics: deck.diagnostics, v6Diagnostics: v6Deck.diagnostics,
    events, verdicts,
  };
}

async function main() {
  const { values } = parseArgs({ options: {
    organizer: { type: 'string' }, scenario: { type: 'string' }, 'approve-plan': { type: 'string' },
  } });
  if (!values.organizer || !values.scenario) throw new Error('expected --organizer <directory> --scenario <Qxx>');
  const directory = fs.realpathSync(values.organizer);
  const fixture = loadPilotCase(directory, values.scenario);
  const { approvalSha256, plan } = fixture;
  if (!values['approve-plan']) {
    const rights = fixture.rightsAssets.map((asset) => ({
      id: asset.id, status: trialImageDecision(fixture.rights, asset),
    }));
    console.log(asJson({ status: 'NOT_RUN', providerCalls: 0, approvalSha256, rights, plan }));
    return;
  }
  if (values['approve-plan'] !== approvalSha256 || process.env.NVIDIA_SPEND_APPROVED !== '1'
      || process.env.NVIDIA_REFERENCE_RIGHTS_CONFIRMED !== '1'
      || !process.env.NVIDIA_API_KEY?.startsWith('nvapi-')) throw new Error('pilot_approval_required');
  if (fixture.rightsAssets.some((asset) => trialImageDecision(fixture.rights, asset) !== 'cleared')) {
    throw new Error('pilot_image_rights_denied');
  }
  // Exclusive creation on the organizer filesystem. Preserve the claim on
  // crash or ambiguous dispatch, and require a new approved plan for reruns.
  const claim = path.join(directory, 'nvidia-' + approvalSha256 + '.json');
  fs.writeFileSync(claim, asJson({ status: 'CLAIMED', approvalSha256, plan }), { flag: 'wx', mode: 0o600 });
  const result = await judgePilotCase(fixture, { apiKey: process.env.NVIDIA_API_KEY });
  const temp = claim + '.tmp';
  fs.writeFileSync(temp, asJson({ status: 'FINISHED', approvalSha256, plan, ...result }), { flag: 'wx', mode: 0o600 });
  fs.renameSync(temp, claim);
  console.log(asJson({
    approvalSha256, providerCalls: result.providerCalls, wallMs: result.wallMs,
    v6VisualPick: result.v6VisualPick, v6Diagnostics: result.v6Diagnostics,
  }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === import.meta.filename) {
  main().catch((error) => {
    console.error(error.code === 'EEXIST' ? 'pilot_already_claimed' : error.message);
    process.exitCode = 2;
  });
}
