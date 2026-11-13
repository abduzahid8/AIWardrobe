/**
 * Read-only category integrity audit of committed merchant snapshots.
 *
 * Usage:
 *   node scripts/stylist6/catalog_category_audit.mjs
 *   node scripts/stylist6/catalog_category_audit.mjs --details
 *
 * No feeds are fetched; no DB updates, merchant API calls or timestamps
 * are invented. "Confirmed" means title AND merchant URL path independently
 * contradict the native catalog category; it never means in-stock.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { assessCatalogCategory } from '../../backend/stylist6/catalog_category_integrity.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export function auditRows(records) {
  const counts = { total: 0, active: 0, confirmed: 0, manualReview: 0, nativeRoleConflict: 0, missingSnapshotObservation: 0 };
  const findings = [];
  const sources = {};
  for (const entry of records) {
    const row = entry.row;
    counts.total += 1;
    if (row?.is_active !== true) continue;
    counts.active += 1;
    if (!row.last_seen_at) counts.missingSnapshotObservation += 1;
    const assessment = assessCatalogCategory(row);
    const status = assessment.status;
    if (status === 'confirmed_category_conflict') counts.confirmed += 1;
    else if (status === 'needs_manual_review') counts.manualReview += 1;
    else if (status === 'native_role_conflict') counts.nativeRoleConflict += 1;
    else continue;
    const source = row.source || entry.file;
    sources[source] ??= { confirmed: 0, manualReview: 0, nativeRoleConflict: 0 };
    const field = status === 'confirmed_category_conflict' ? 'confirmed'
      : status === 'needs_manual_review' ? 'manualReview' : 'nativeRoleConflict';
    sources[source][field] += 1;
    findings.push({
      id: row.id, name: row.name, source, file: entry.file,
      status, nativeCategory: row.category,
      suggested: assessment.suggested ?? null,
    });
  }
  return {
    counts: {
      ...counts,
      uniqueConfirmedSkuIds: new Set(findings.filter(x =>
        x.status === 'confirmed_category_conflict').map(x => x.id)).size,
    },
    sources, findings,
  };
}

export function readLocalCatalog({ directory = path.join(ROOT, 'data') } = {}) {
  const entries = [];
  const files = fs.readdirSync(directory).filter(name =>
    name.startsWith('shop_catalog_') && name.endsWith('.json')).sort();
  for (const file of files) {
    const json = JSON.parse(fs.readFileSync(path.join(directory, file), 'utf8'));
    const rows = Array.isArray(json) ? json : (Array.isArray(json.items) ? json.items : json.products);
    if (!Array.isArray(rows)) throw new Error('invalid_catalog_snapshot: ' + file);
    for (const row of rows) entries.push({ file, row });
  }
  return { files: files.length, entries };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.some(arg => !['--details'].includes(arg) && arg.startsWith('--'))) {
    throw new Error('unsupported_option');
  }
  const { files, entries } = readLocalCatalog();
  const audit = auditRows(entries);
  const report = {
    sourceFiles: files, ...audit.counts,
    findingsBySource: Object.entries(audit.sources)
      .map(([source, count]) => ({ source, ...count }))
      .sort((a, b) => b.confirmed - a.confirmed).slice(0, 30),
    ...(process.argv.includes('--details') ? { confirmedFindings: audit.findings
      .filter(x => x.status === 'confirmed_category_conflict').slice(0, 150) } : {}),
    note: 'Read-only historical JSON snapshots; last_seen_at absent here does not imply the LIVE database has no observation. A URL is not stock, size or delivery verification.',
  };
  console.log(JSON.stringify(report, null, 2));
}
