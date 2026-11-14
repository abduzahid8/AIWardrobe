#!/usr/bin/env node
// Scan tracked and non-ignored untracked files, including JSON. Never print values.
import { execFileSync } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
const files = [
  ...new Set(
    execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], {
      maxBuffer: 32 * 1024 * 1024,
    })
      .toString()
      .split('\0')
      .filter(Boolean)
  ),
];
const patterns = [
  ['google_key', /AIza[0-9A-Za-z_-]{35}/g],
  ['provider_secret', /(?:sk-ant-|sk-proj-|sb_secret_)[0-9A-Za-z_-]{20,}/g],
  ['slack_token', /xox[bpas]-[0-9A-Za-z-]{10,}/g],
  ['private_key', /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/g],
];
let hits = 0,
  scanned = 0;
for (const file of files) {
  let body;
  try {
    if (!statSync(file).isFile()) continue;
    body = readFileSync(file);
  } catch {
    continue;
  }
  if (body.includes(0)) continue;
  const source = body.toString('utf8');
  scanned++;
  for (const [name, pattern] of patterns) {
    pattern.lastIndex = 0;
    if (pattern.test(source)) {
      console.log(`HIT ${name} ${file}`);
      hits++;
    }
  }
  for (const token of source.match(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g) || []) {
    try {
      if (
        JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString()).role === 'service_role'
      ) {
        console.log(`HIT service_role_jwt ${file}`);
        hits++;
        break;
      }
    } catch {
      /* Not a JWT; no value is logged. */
    }
  }
}
console.log(
  `SECRET_SCAN ${hits ? 'FAIL' : 'PASS'} (${scanned} text files; ${hits} findings; worktree only, ignored files/history/binaries excluded)`
);
process.exitCode = hits ? 1 : 0;
