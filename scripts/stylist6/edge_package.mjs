/** Reproducible, secret-free deployment descriptor; source is never copied by hand. */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
export async function buildEdgePackage() {
  const names = [
    'backend/supabase/functions/aiw-stylist6-node-staging/index.ts',
    'backend/supabase/functions/aiw-stylist6-node-staging/deno.json',
    'backend/stylist-proxy/stylist6-staging.ts',
    'backend/stylist-node/transport/signature.mjs',
  ];
  return {
    project_id: 'fyqpifmrsftsfqibhwhy', name: 'aiw-stylist6-node-staging',
    entrypoint_path: names[0], verify_jwt: true,
    files: await Promise.all(names.map(async name => ({name, content:await readFile(resolve(root,name),'utf8')}))),
  };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!process.argv[2]) throw new Error('Usage: node scripts/stylist6/edge_package.mjs /tmp/edge-package.json');
  await writeFile(process.argv[2],JSON.stringify(await buildEdgePackage(),null,2)+'\n',{mode:0o600});
  console.log('Secret-free Stylist 6 deployment descriptor written');
}
