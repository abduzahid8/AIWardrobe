import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import ts from 'typescript';

// Execute the real store helper in separate processes: each Date implementation
// must use the target timezone, including its daylight-saving transitions.
const file = path.resolve(__dirname, '../../store/subscriptionStore.ts');
const source = fs.readFileSync(file, 'utf8');
const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
const helper = ast.statements.find(s => ts.isFunctionDeclaration(s) && s.name?.text === 'computeTrialState');
if (!helper) throw new Error('Trial helper missing');
const js = ts.transpileModule(`const FREE_TRIAL_DAYS = 7;\n${helper.getText(ast)}`, {
  compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS },
}).outputText;

const hookFile = path.resolve(__dirname, '../../hooks/useTrialStatus.ts');
const hookAst = ts.createSourceFile(hookFile, fs.readFileSync(hookFile, 'utf8'), ts.ScriptTarget.Latest, true);
const statusHelper = hookAst.statements.find(s => ts.isFunctionDeclaration(s) && s.name?.text === 'buildStatus');
if (!statusHelper) throw new Error('Trial display helper missing');
const displayJs = ts.transpileModule(statusHelper.getText(hookAst).replace(
  "require('../store/subscriptionStore')", '({ FREE_TRIAL_DAYS: 7 })',
), { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText;

test.each<[string, string, string, boolean]>([
  ['America/New_York', '2026-03-06T12:00:00Z', '2026-03-13T11:30:00Z', true],
  ['America/New_York', '2026-10-30T12:00:00Z', '2026-11-06T12:30:00Z', false],
  ['Europe/Berlin', '2026-03-26T12:00:00Z', '2026-04-02T11:30:00Z', true],
  ['Europe/Berlin', '2026-10-23T12:00:00Z', '2026-10-30T12:30:00Z', false],
  ['Asia/Tashkent', '2026-03-06T12:00:00Z', '2026-03-13T11:30:00Z', true],
  ['Asia/Tashkent', '2026-03-06T12:00:00Z', '2026-03-13T12:00:00Z', false],
])('seven-day trial in %s from %s at %s', (zone, start, now, active) => {
  const script = `const RealDate = Date;
    global.Date = class extends RealDate {
      constructor(...args) { super(...(args.length ? args : [${JSON.stringify(now)}])); }
      static now() { return RealDate.parse(${JSON.stringify(now)}); }
    };
    ${js}
    ${displayJs}
    const trial = computeTrialState('free', ${JSON.stringify(start)});
    process.stdout.write(JSON.stringify({ ...trial, display: buildStatus({ ...trial, trialStartedAt: ${JSON.stringify(start)} }) }));`;
  const result = JSON.parse(execFileSync(process.execPath, ['-e', script], {
    encoding: 'utf8', env: { ...process.env, TZ: zone },
  }));
  expect(result.isTrialActive).toBe(active);
  expect(result.isTrialExpired).toBe(!active);
  expect(result.trialDaysRemaining).toBe(active ? 1 : 0);
  expect(result.display.hoursRemaining).toBe(active ? 1 : 0);
});
