#!/usr/bin/env node
/** Local-only release evidence. No deployment, hosted SQL, provider call or env loading. */
import { execFileSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('..', import.meta.url));
process.chdir(root);
const out = resolve('docs/launch/evidence');
mkdirSync(out, { recursive: true });
const json = (file) => JSON.parse(readFileSync(file, 'utf8'));
const sha = (body) => createHash('sha256').update(body).digest('hex');
const git = (...args) =>
  execFileSync('git', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).trim();
const files = git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
  .split('\0')
  .filter(Boolean);
const snapshot = () =>
  [
    ...new Set(
      git('ls-files', '--cached', '--others', '--exclude-standard', '-z')
        .split('\0')
        .filter(Boolean)
    ),
  ]
    .sort()
    .filter(
      (file) =>
        !file.startsWith('docs/launch/evidence/') && !file.startsWith('docs/launch/generated-')
    )
    .flatMap((file) => {
      try {
        return statSync(file).isFile() ? [{ file, sha256: sha(readFileSync(file)) }] : [];
      } catch {
        return [];
      }
    });
const initialSnapshot = snapshot(),
  sourceHash = sha(JSON.stringify(initialSnapshot));
const checks = [];
async function run(check, command, args, options = {}) {
  const log = resolve(out, `${check}.log`),
    started = Date.now();
  console.log(`Running ${check}...`);
  const stream = createWriteStream(log);
  const child = spawn(command, args, {
    cwd: root,
    env: { ...process.env, ...options.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.pipe(stream, { end: false });
  child.stderr.pipe(stream, { end: false });
  const timer = setTimeout(() => child.kill('SIGTERM'), options.timeout || 600000);
  const result = await new Promise((done) => {
    child.on('error', (error) => done({ code: null, error: String(error) }));
    child.on('close', (code, signal) => done({ code, signal }));
  });
  clearTimeout(timer);
  await new Promise((done) => stream.end(done));
  checks.push({
    check,
    result: result.code === 0 ? 'PASS' : 'FAIL',
    command: [command, ...args],
    exitCode: result.code,
    signal: result.signal || null,
    error: result.error || null,
    elapsedMs: Date.now() - started,
    log: log.replace(root, ''),
  });
  console.log(`${check}: ${checks.at(-1).result}`);
}
const manual = (check, detail) => checks.push({ check, result: 'MANUAL REQUIRED', detail });
const pkg = json('package.json'),
  app = json('app.json').expo;
const migrationNames = json('docs/launch/activation-config.json').pendingMigrations;
const migrationSet = migrationNames.map((file) => ({
  file,
  sha256: sha(readFileSync(`backend/supabase/migrations/${file}`)),
}));
const edgeRoot = 'backend/supabase/functions';
const edgeFiles = files.filter(
  (file) => file.startsWith(`${edgeRoot}/`) && /\.(ts|json|lock)$/.test(file)
);
const functions = readdirSync(edgeRoot)
  .filter((name) => !name.startsWith('_') && statSync(`${edgeRoot}/${name}`).isDirectory())
  .sort();
const flagSource = readFileSync('src/services/platform/experimentation.ts', 'utf8');
const flagBlock = flagSource.match(/PLATFORM_FLAG_DEFAULTS[^=]*=\s*\{([\s\S]*?)\};/)?.[1];
if (!flagBlock) throw new Error('Cannot extract actual platform flag defaults');
const actualFlags = Object.fromEntries(
  [...flagBlock.matchAll(/(\w+):\s*(true|false)/g)].map(([, key, value]) => [key, value === 'true'])
);
const manifest = {
  schemaVersion: 1,
  version: pkg.version,
  buildNumber: { ios: app.ios.buildNumber, android: app.android.versionCode },
  gitCommit: git('rev-parse', 'HEAD'),
  dirty: !!git('status', '--porcelain'),
  worktreeSha256: sourceHash,
  localMigrationSet: migrationSet,
  localEdgeFunctionSet: functions.map((name) => ({
    name,
    entrySha256: sha(readFileSync(`${edgeRoot}/${name}/index.ts`)),
  })),
  edgeTreeSha256: sha(
    JSON.stringify(edgeFiles.sort().map((file) => ({ file, sha256: sha(readFileSync(file)) })))
  ),
  featureFlagDefaults: {
    platform: actualFlags,
    core: {
      wardrobe: true,
      outfitGeneration: true,
      buyCheck: true,
      fitFeedback: true,
      familyVariantUI: true,
    },
    serverPaidWorkControls: {
      globalUnsetDefault: false,
      disabledEndpointsUnsetDefault: [],
      preparedActivation: json('docs/launch/activation-config.json').serverPaidWorkControls,
      hostedValues: 'UNVERIFIED; not changed',
    },
    remoteKillSwitchDrill: 'MANUAL REQUIRED',
  },
  entitlementConfigVersion: {
    matrix: 1,
    sourceSha256: sha(readFileSync('src/services/growth/entitlements.ts')),
    revenuecatMappingSha256: sha(
      readFileSync('backend/supabase/functions/_shared/revenuecatSnapshot.ts')
    ),
    hostedConfig: 'UNVERIFIED',
  },
  promptSchemaVersions: {
    styleEvidence: 1,
    visualCache: 'v1',
    decisionEvent: 1,
    betaFeedback: 1,
    promptBundleSha256: sha(readFileSync('backend/supabase/functions/generate-outfits/index.ts')),
  },
  knownLimitations: ['docs/KNOWN_LIMITATIONS.md', 'docs/launch/KNOWN_LIMITATIONS.md'],
  blockers: json('docs/launch/blockers.json'),
  manualQAStatus: 'PENDING — docs/launch/BETA_QA.md',
  remoteActivationStatus: 'PENDING EXPLICIT APPROVAL; nothing activated',
  createdAt: new Date().toISOString(),
};
await run('typecheck', 'npm', ['run', 'typecheck']);
await run('lint', 'npm', ['run', 'lint']);
const denoPath = existsSync(resolve(homedir(), '.deno/bin/deno'))
  ? resolve(homedir(), '.deno/bin/deno')
  : 'deno';
await run('edge_typecheck', denoPath, [
  'check',
  '--no-config',
  '--node-modules-dir=none',
  '--lock=backend/supabase/functions/deno.lock',
  '--frozen',
  ...functions.map((name) => `${edgeRoot}/${name}/index.ts`),
]);
await run(
  'ios_bundle',
  'node',
  [
    'node_modules/expo/bin/cli',
    'export',
    '--platform',
    'ios',
    '--output-dir',
    '/tmp/aiwardrobe-rc-ios-export',
    '--max-workers',
    '2',
  ],
  { env: { EXPO_NO_DOTENV: '1', EXPO_OFFLINE: '1' } }
);
await run(
  'android_bundle',
  'node',
  [
    'node_modules/expo/bin/cli',
    'export',
    '--platform',
    'android',
    '--output-dir',
    '/tmp/aiwardrobe-rc-android-export',
    '--max-workers',
    '2',
  ],
  { env: { EXPO_NO_DOTENV: '1', EXPO_OFFLINE: '1' } }
);
await run('critical_tests', 'node', [
  'node_modules/jest/bin/jest.js',
  '__tests__/launch',
  '__tests__/growth',
  '__tests__/platform',
  '__tests__/purchase',
  '__tests__/components/betaFeedback.test.tsx',
  '__tests__/backend/rateLimit.test.js',
  '__tests__/backend/paidWorkCoverage.test.js',
  '__tests__/backend/tryOnContract.test.js',
  '__tests__/services/outfitGenerationFailure.test.ts',
  '__tests__/features/closestWardrobeOutfits.test.ts',
  '--runInBand',
  '--silent',
]);
await run(
  'full_regressions',
  'node',
  ['node_modules/jest/bin/jest.js', '--runInBand', '--silent'],
  { env: { AIWARDROBE_WRITE_STRESS_REPORT: '1' } }
);
await run('migration_tests', 'node', [
  'scripts/validation/catalog-discovery-local.mjs',
  '--release',
]);
await run('security_checks', 'node', ['--test', 'scripts/validation/dependency-security.test.mjs']);
await run('secrets_scan', 'node', ['scripts/security/secret-scan.mjs']);
await run('feature_flags_localization', 'node', ['scripts/launch/static-checks.mjs']);
await run('compatibility_contracts', 'node', [
  'node_modules/jest/bin/jest.js',
  '__tests__/services/catalogDiscovery.test.ts',
  '__tests__/services/catalogRepository.test.ts',
  '__tests__/launch/launch.test.ts',
  '__tests__/services/iapService.test.ts',
  '--runInBand',
  '--silent',
]);
await run('accessibility_feedback', 'node', [
  'node_modules/jest/bin/jest.js',
  '__tests__/components/betaFeedback.test.tsx',
  '--runInBand',
  '--silent',
]);
await run('beta_review_tools', 'node', ['--test', 'scripts/launch/beta-review.test.mjs']);
manual(
  'accessibility_device',
  'VoiceOver/TalkBack, text scaling, screen-reader result order and contrast on four required devices pending'
);
manual(
  'compatibility_installed_binaries',
  'A/B/C/D backend + installed old/new client matrix not proven by contract tests'
);
manual(
  'migration_hosted_baseline',
  'Pending set tested against bounded fixtures, not hosted schema. Approve schema comparison/backup rehearsal separately.'
);

manual(
  'device_qa_performance',
  'Four required devices, native image imports/scroll/memory/background, signed release build and sandbox billing pending'
);
manual(
  'remote_activation',
  'Ordered packet: docs/launch/REMOTE_ACTIVATION.md; all remote mutations require explicit approval'
);
manual(
  'secret_history_rotation',
  'Current text worktree scanned. History/bundles and rotation of historical credential require review; see blockers.'
);
const finalHash = sha(JSON.stringify(snapshot()));
checks.push({
  check: 'source_unchanged_during_check',
  result: sourceHash === finalHash ? 'PASS' : 'FAIL',
  detail: 'Worktree fingerprint excludes generated evidence only',
});
const openBlockers = manifest.blockers.filter(
  (b) => b.severity === 'BLOCKER' && b.fixStatus !== 'closed'
);
const result = checks.some((c) => c.result === 'FAIL')
  ? 'FAIL'
  : openBlockers.length || checks.some((c) => c.result === 'MANUAL REQUIRED')
    ? 'MANUAL REQUIRED'
    : 'PASS';
const report = {
  schemaVersion: 1,
  result,
  decision:
    result === 'PASS' ? 'RELEASE CANDIDATE READY FOR APPROVAL' : 'RELEASE CANDIDATE BLOCKED',
  manifest,
  checks,
  remoteActivation: 'PENDING EXPLICIT APPROVAL',
};
writeFileSync(resolve(out, 'release-candidate.json'), JSON.stringify(report, null, 2) + '\n');
writeFileSync(
  resolve(out, 'source-snapshot.json'),
  JSON.stringify(initialSnapshot, null, 2) + '\n'
);
writeFileSync(
  resolve(out, 'release-candidate.md'),
  `# Local release candidate report\n\n${report.decision}\n\nVersion ${manifest.version}; iOS ${manifest.buildNumber.ios}; Android ${manifest.buildNumber.android}. Base commit ${manifest.gitCommit}; dirty=${manifest.dirty}; worktree SHA-256 ${sourceHash}.\n\n| Check | Result | Evidence |\n|---|---|---|\n${checks.map((c) => `| ${c.check} | ${c.result} | ${c.log || c.detail} |`).join('\n')}\n\nREMOTE ACTIVATION PENDING EXPLICIT APPROVAL\n`
);
console.log(
  JSON.stringify(
    {
      result,
      decision: report.decision,
      report: 'docs/launch/evidence/release-candidate.json',
      checks: checks.map(({ check, result }) => ({ check, result })),
    },
    null,
    2
  )
);
// Nonzero means the complete release gate is not satisfied, even when local tests pass.
process.exitCode = result === 'PASS' ? 0 : result === 'FAIL' ? 1 : 2;
