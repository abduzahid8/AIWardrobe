// Sync all three locale files to have identical key structure
const fs = require('fs');
const path = require('path');

const enPath = path.join(__dirname, 'i18n', 'locales', 'en.json');
const ruPath = path.join(__dirname, 'i18n', 'locales', 'ru.json');
const uzPath = path.join(__dirname, 'i18n', 'locales', 'uz.json');

const en = JSON.parse(fs.readFileSync(enPath, 'utf8'));
const ru = JSON.parse(fs.readFileSync(ruPath, 'utf8'));
const uz = JSON.parse(fs.readFileSync(uzPath, 'utf8'));

function getAllKeys(obj, prefix = '') {
  const keys = [];
  for (const [k, v] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      keys.push(...getAllKeys(v, fullKey));
    } else {
      keys.push(fullKey);
    }
  }
  return keys;
}

function getValue(obj, keyPath) {
  const parts = keyPath.split('.');
  let current = obj;
  for (const p of parts) {
    if (current === undefined || current === null) return undefined;
    if (Array.isArray(current)) {
      const idx = parseInt(p);
      if (isNaN(idx)) return undefined;
      current = current[idx];
    } else {
      current = current[p];
    }
  }
  return current;
}

function setValue(obj, keyPath, value) {
  const parts = keyPath.split('.');
  let current = obj;
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    const nextIdx = parseInt(parts[i + 1]);
    if (i === parts.length - 1) {
      current[p] = value;
    } else {
      if (!isNaN(nextIdx)) {
        if (!current[p]) current[p] = [];
        current = current[p];
      } else {
        if (!current[p] || typeof current[p] !== 'object') current[p] = {};
        current = current[p];
      }
    }
  }
}

// 1. Find keys in EN that are missing from RU/UZ and add them
// 2. Find keys in RU/UZ that are missing from EN and add them

const enKeys = getAllKeys(en);
const ruKeys = getAllKeys(ru);
const uzKeys = getAllKeys(uz);

const ruKeySet = new Set(ruKeys);
const uzKeySet = new Set(uzKeys);
const enKeySet = new Set(enKeys);

// Add missing EN->RU
let addedRu = 0;
for (const key of enKeys) {
  if (!ruKeySet.has(key)) {
    const val = getValue(en, key);
    setValue(ru, key, `[RU] ${val}`);
    addedRu++;
  }
}

// Add missing EN->UZ
let addedUz = 0;
for (const key of enKeys) {
  if (!uzKeySet.has(key)) {
    const val = getValue(en, key);
    setValue(uz, key, `[UZ] ${val}`);
    addedUz++;
  }
}

// Add missing RU->EN
let addedEn = 0;
for (const key of ruKeys) {
  if (!enKeySet.has(key)) {
    const val = getValue(ru, key);
    setValue(en, key, `[EN] ${val}`);
    addedEn++;
  }
}

// Add missing UZ->EN (uniquely uz)
for (const key of uzKeys) {
  if (!enKeySet.has(key)) {
    const val = getValue(uz, key);
    setValue(en, key, `[EN] ${val}`);
    addedEn++;
  }
}

// Write back
fs.writeFileSync(enPath, JSON.stringify(en, null, 2) + '\n');
fs.writeFileSync(ruPath, JSON.stringify(ru, null, 2) + '\n');
fs.writeFileSync(uzPath, JSON.stringify(uz, null, 2) + '\n');

console.log(`Added ${addedRu} keys to ru.json`);
console.log(`Added ${addedUz} keys to uz.json`);
console.log(`Added ${addedEn} keys to en.json`);

// Print newly added keys that need real translations
console.log('\n=== Keys added to ru.json that need proper RU translations ===');
for (const key of enKeys) {
  if (!ruKeySet.has(key)) {
    console.log(`  ${key}: ${JSON.stringify(getValue(en, key))}`);
  }
}

console.log('\n=== Keys added to uz.json that need proper UZ translations ===');
for (const key of enKeys) {
  if (!uzKeySet.has(key)) {
    console.log(`  ${key}: ${JSON.stringify(getValue(en, key))}`);
  }
}