/** Execute app media source in Node. Native staging is deliberately unavailable. */
import { decode, encode } from 'base64-arraybuffer';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { transpileModule, ModuleKind } from 'typescript';
export async function loadAppMediaForNode(client, events, base) {
  const hashes = {};
  async function execute(name, dependencies) {
    const code = transpileModule(await readFile(new URL(`../../src/services/${name}.ts`, import.meta.url),'utf8'),
      {compilerOptions:{module:ModuleKind.CommonJS}}).outputText;
    hashes[name] = createHash('sha256').update(code).digest('hex');
    const exports = {};
    new Function('exports','require',code)(exports, name => {
      if (!(name in dependencies)) throw new Error('Unexpected app media dependency');
      return dependencies[name];
    });
    return exports;
  }
  const deadline = await execute('../lib/requestDeadline', {});
  const media = await execute('wardrobeMedia', {'base64-arraybuffer':{decode}, '../../lib/supabase':{supabase:client},
    '../../store/authEvents':events, '../config/env':{Config:{supabase:{url:base}}}, '../lib/requestDeadline':deadline});
  const pending = await execute('pendingWardrobeMedia', {'expo-file-system/legacy':{documentDirectory:null},
    'base64-arraybuffer':{encode}, '../../store/authEvents':events, './wardrobeMedia':media});
  const itemMedia = await execute('wardrobeItemMedia', {'./wardrobeMedia':media, './pendingWardrobeMedia':pending});
  return {media,itemMedia,hashes};
}
