/** Real OS copy/restart probe through an Expo-shaped adapter; no native codec or network claim. */
import { readFile, writeFile, mkdir, stat, rename, rm, mkdtemp, copyFile, readdir, truncate } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { transpileModule, ModuleKind } from 'typescript';
import assert from 'node:assert/strict';
const owner = '11111111-1111-4111-8111-111111111111', other = '22222222-2222-4222-8222-222222222222';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
async function load(documents) {
  const compile = async path => transpileModule(await readFile(new URL(path, import.meta.url), 'utf8'), { compilerOptions: { module: ModuleKind.CommonJS } }).outputText;
  const events = {}, media = {};
  new Function('exports', await compile('../../store/authEvents.ts'))(events);
  const fs = {
    writeAsStringAsync: (uri, contents) => writeFile(fileURLToPath(uri), contents),
    readDirectoryAsync: uri => readdir(fileURLToPath(uri)),
    documentDirectory: pathToFileURL(documents).href + '/',
    getInfoAsync: async uri => { try { const s = await stat(fileURLToPath(uri)); return { exists: true, isDirectory: s.isDirectory(), size: s.size }; } catch(e) { if (e.code === 'ENOENT') return { exists: false }; throw e; } },
    makeDirectoryAsync: uri => mkdir(fileURLToPath(uri), { recursive: true }),
    copyAsync: ({from,to}) => copyFile(fileURLToPath(from),fileURLToPath(to)),
    moveAsync: ({from,to}) => rename(fileURLToPath(from),fileURLToPath(to)),
    deleteAsync: uri => rm(fileURLToPath(uri), { recursive: true, force: true }),
  };
  const code = await compile('../../src/services/pendingScanMedia.ts');
  new Function('exports','require',code)(media, name => {
    if (name === 'expo-file-system/legacy') return fs;
    if (name === '../../store/authEvents') return events;
    throw new Error('Unexpected dependency');
  });
  events.authEvents.emitLogin(owner);
  return { media, fs, events, codeSha256: hash(code) };
}
if (process.argv[2] === '--restore') {
  const root = process.argv[3], manifest = JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
  const { media } = await load(join(root,'new-container','Documents'));
  for (const item of manifest) {
    await assert.rejects(stat(fileURLToPath(item.source)));
    await assert.rejects(stat(fileURLToPath(item.staged)));
    const resolved = media.resolvePendingScanMedia(item.staged,owner);
    assert.equal(hash(await readFile(fileURLToPath(resolved))),item.sha256);
    assert.equal(await media.stagePendingScanMedia(owner,item.staged,item.type),resolved);
    assert.throws(() => media.resolvePendingScanMedia(item.staged,other));
    await media.removePendingScanMedia(other,item.staged);
    assert.equal(hash(await readFile(fileURLToPath(resolved))),item.sha256);
    await media.removePendingScanMedia(owner,item.staged);
    await assert.rejects(stat(fileURLToPath(resolved)));
  }
  process.stdout.write(JSON.stringify({ processRestart: true, originalSourcesEvicted: true, oldContainerAbsent: true, restoredByteHashesMatch: true, imageAndVideoCopies: 2, otherOwnerBlocked: true, managedCleanup: true })+'\n');
} else {
  const root = await mkdtemp(join(tmpdir(),'aiwardrobe-pending-scan-'));
  try {
    const oldDocuments = join(root,'old-container','Documents');
    const {media,fs,events,codeSha256} = await load(oldDocuments);
    const sources = [
      { name:'capture.jpg', type:'image', bytes: await readFile(new URL('../../docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg',import.meta.url)) },
      { name:'capture.mp4', type:'video', bytes: Buffer.alloc(1024*1024,0x42) }, // Synthetic copy payload, not a video decode fixture.
    ];
    const manifest = [];
    for (const item of sources) {
      const path = join(root,item.name); await writeFile(path,item.bytes);
      const source = pathToFileURL(path).href;
      const staged = await media.stagePendingScanMedia(owner,source,item.type);
      assert.equal(hash(await readFile(fileURLToPath(staged))),hash(item.bytes));
      manifest.push({type:item.type,source,staged,sha256:hash(item.bytes),bytes:item.bytes.length});
    }
    const folder = join(oldDocuments,'aiwardrobe-scan-queue',owner);
    const before = (await readdir(folder)).length;
    const originalCopy = fs.copyAsync;
    fs.copyAsync = async ({to}) => writeFile(fileURLToPath(to),'short');
    await assert.rejects(media.stagePendingScanMedia(owner,manifest[0].source,'image'),/copy_incomplete/);
    assert.equal((await readdir(folder)).length,before);
    fs.copyAsync = async options => { await originalCopy(options); events.authEvents.emitLogin(other); };
    await assert.rejects(media.stagePendingScanMedia(owner,manifest[0].source,'image'),/account_changed/);
    assert.equal((await readdir(folder)).length,before);
    events.authEvents.emitLogin(owner); fs.copyAsync = originalCopy;
    const large = join(root,'large.mp4'); await writeFile(large,''); await truncate(large,100*1024*1024+1);
    let copies = 0; fs.copyAsync = async options => { copies++; await originalCopy(options); };
    await assert.rejects(media.stagePendingScanMedia(owner,pathToFileURL(large).href,'video'),/invalid_size/); assert.equal(copies,0);
    await writeFile(join(root,'manifest.json'),JSON.stringify(manifest));
    for (const item of manifest) await rm(fileURLToPath(item.source));
    await mkdir(join(root,'new-container'),{recursive:true}); await rename(oldDocuments,join(root,'new-container','Documents'));
    const child = await promisify(execFile)(process.execPath,[fileURLToPath(import.meta.url),'--restore',root],{timeout:30000});
    const result = { measuredAt:new Date().toISOString(), codeSha256, ...JSON.parse(child.stdout), partialCopyRejected:true, accountSwitchDuringCopyRejected:true, oversizeRejectedBeforeCopy:true, fixtures:manifest.map(({type,sha256,bytes})=>({type,sha256,bytes})), scope:'Actual filesystem and child-process restart through Expo adapter; synthetic video bytes are not video-decoder or device evidence' };
    await writeFile(new URL('../../docs/reports/readiness-evidence/pending-scan-durability.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
    process.stdout.write(JSON.stringify(result)+'\n');
  } finally { await rm(root,{recursive:true,force:true}); }
}
