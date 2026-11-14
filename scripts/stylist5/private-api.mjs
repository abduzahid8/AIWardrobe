import fs from 'node:fs';
export const PROJECT = 'fyqpifmrsftsfqibhwhy';
export function token() {
  if (process.env.SUPABASE_ACCESS_TOKEN) return process.env.SUPABASE_ACCESS_TOKEN;
  if (!process.env.SUPABASE_ACCESS_TOKEN_FILE) throw Error('Set SUPABASE_ACCESS_TOKEN_FILE to the approved local CLI credential file');
  return fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE, 'utf8').trim();
}
export async function management(path, init = {}) {
  const response = await fetch(`https://api.supabase.com/v1/projects/${PROJECT}${path}`, {
    ...init, headers: {Authorization: `Bearer ${token()}`, ...init.headers}, signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) throw Error(`Supabase management HTTP ${response.status}`);
  return response;
}
export async function query(sql) {
  return (await management('/database/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:sql})})).json();
}
export const quote = value => "'" + String(value).replaceAll("'", "''") + "'";
