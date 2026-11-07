/** Frozen local pre-mission evaluator, excluded from production runtime. */
import { readFileSync } from 'node:fs';
import ts from 'typescript';
const snapshot = JSON.parse(
  readFileSync('scripts/validation/stylist-quality-baseline.json', 'utf8')
);
const cache: Record<string, any> = {};
export function baselineModule(name: string): any {
  if (cache[name]) return cache[name];
  const module = { exports: {} as any };
  cache[name] = module.exports;
  const source = snapshot.sources[name];
  if (typeof source !== 'string') throw Error('unknown_baseline_module');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  new Function('exports', 'module', 'require', compiled)(module.exports, module, (path: string) =>
    baselineModule(path.replace(/^\.\//, '').replace(/\.ts$/, ''))
  );
  cache[name] = module.exports;
  return module.exports;
}
