// One-command local startup for the Stylist 6 development loop:
// validates config (never prints secrets), starts the dev bridge, waits for
// readiness, then starts Metro. Ctrl-C stops both. Dev only, never prod.
import { spawn, execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isCurrentStylistBridgeHealth, STYLIST6_DEV_ENGINE_VERSION } from './engine_identity.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

const fail = (message) => {
  console.error(`[dev:stylist6] ERROR: ${message}`);
  process.exit(1);
};

// Config validation: presence only, values never printed.
const envFile = path.join(root, '.env');
if (!fs.existsSync(envFile)) fail('missing .env (copy .env.example and fill local values)');
const envText = fs.readFileSync(envFile, 'utf8');
const has = (key) => new RegExp(`^${key}=.+`, 'm').test(envText);
for (const key of ['EXPO_PUBLIC_SUPABASE_URL', 'EXPO_PUBLIC_SUPABASE_ANON_KEY']) {
  if (!has(key)) fail(`.env is missing ${key} (catalog hydration and inventory need it)`);
}
const port = Number(args.port || process.env.STYLIST6_DEV_PORT || 3127);
if (!Number.isInteger(port) || port <= 0) fail(`invalid port ${args.port || ''}`);
const bridgeFlag = /^EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true/m.test(envText);
console.log(`[dev:stylist6] config ok (supabase vars present, stylist6 dev bridge ${bridgeFlag ? 'enabled' : 'DISABLED in .env'})`);
const nvidiaPresent = /^NVIDIA_API_KEY=nvapi-.+/m.test(envText);
console.log(`[dev:stylist6] nvidia visual review ${nvidiaPresent ? 'available (key present)' : 'unavailable (no key; decks stay NVIDIA_NOT_RUN)'}`);
if (!bridgeFlag) {
  console.log('[dev:stylist6] hint: set EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true in .env to show the Stylist 6 picker');
}

const healthUrl = `http://127.0.0.1:${port}/health`;
const healthy = async () => {
  try {
    const res = await fetch(healthUrl, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return false;
    const body = await res.json();
    return isCurrentStylistBridgeHealth(body);
  } catch {
    return false;
  }
};
const portInUse = () => {
  try {
    const out = execSync(`lsof -ti tcp:${port} 2>/dev/null`, { encoding: 'utf8' }).trim();
    return out.split('\n').filter(Boolean);
  } catch {
    return [];
  }
};

let bridge = null;
if (await healthy()) {
  console.log(`[dev:stylist6] current bridge ${STYLIST6_DEV_ENGINE_VERSION} healthy on :${port}, reusing it`);
} else {
  const pids = portInUse();
  if (pids.length) {
    // A stale listener that is not our bridge blocks the port. Only remove
    // processes that are verifiably a previous dev bridge.
    let cleared = false;
    for (const pid of pids) {
      try {
        const cmd = execSync(`ps -p ${pid} -o command=`, { encoding: 'utf8' });
        if (/dev_bridge\.mjs/.test(cmd)) {
          console.log(`[dev:stylist6] stopping stale bridge (pid ${pid})`);
          process.kill(Number(pid), 'SIGTERM');
          cleared = true;
        }
      } catch { /* not ours to touch */ }
    }
    if (!cleared) fail(`port ${port} is occupied by a non-bridge process (pids: ${pids.join(',')}); free it and retry`);
    await new Promise((resolve) => setTimeout(resolve, 1500));
    if (await healthy()) {
      console.log(`[dev:stylist6] stale bridge cleared; a new one is already healthy on :${port}`);
    }
  }
  if (!(await healthy())) {
    console.log(`[dev:stylist6] starting bridge on :${port}`);
    bridge = spawn('node', ['scripts/stylist6/dev_bridge.mjs', `--port=${port}`], {
      cwd: root, stdio: 'inherit',
    });
    bridge.on('exit', (code) => {
      if (code !== 0 && code !== null) console.error(`[dev:stylist6] bridge exited with code ${code}`);
    });
    const deadline = Date.now() + 20000;
    for (;;) {
      if (await healthy()) break;
      if (Date.now() > deadline) fail('bridge did not become healthy in 20s; see bridge output above');
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    console.log(`[dev:stylist6] bridge ready on :${port}`);
  }
}

console.log('[dev:stylist6] starting Metro (expo start --ios). Ctrl-C stops Metro and the bridge.');
const expoArgs = ['expo', 'start', '--ios'];
const dashDash = process.argv.indexOf('--');
if (dashDash >= 0) expoArgs.push(...process.argv.slice(dashDash + 1));
const metro = spawn('npx', expoArgs, { cwd: root, stdio: 'inherit', shell: false });

const shutdown = (signal) => {
  console.log(`[dev:stylist6] ${signal} received, stopping...`);
  try { metro.kill(signal === 'SIGINT' ? 'SIGINT' : 'SIGTERM'); } catch { /* already gone */ }
  if (bridge) {
    try { bridge.kill('SIGTERM'); } catch { /* already gone */ }
  }
  setTimeout(() => process.exit(0), 1500).unref();
};
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
metro.on('exit', (code) => {
  if (bridge) {
    try { bridge.kill('SIGTERM'); } catch { /* already gone */ }
  }
  process.exit(code ?? 0);
});
