const fs = require('fs');
const path = require('path');

// Read JSON files
const en = JSON.parse(fs.readFileSync(path.join(__dirname, 'i18n/locales/en.json'), 'utf8'));
const ru = JSON.parse(fs.readFileSync(path.join(__dirname, 'i18n/locales/ru.json'), 'utf8'));
const uz = JSON.parse(fs.readFileSync(path.join(__dirname, 'i18n/locales/uz.json'), 'utf8'));

// Function to get all keys recursively
function getAllKeys(obj, prefix = '') {
  const keys = [];
  for (const key in obj) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (typeof obj[key] === 'object' && obj[key] !== null) {
      keys.push(...getAllKeys(obj[key], fullKey));
    } else {
      keys.push(fullKey);
    }
  }
  return keys;
}

const enKeys = new Set(getAllKeys(en));
const ruKeys = new Set(getAllKeys(ru));
const uzKeys = new Set(getAllKeys(uz));

// Find keys missing in each language
const missingInRu = [...enKeys].filter(k => !ruKeys.has(k));
const missingInUz = [...enKeys].filter(k => !uzKeys.has(k));
const extraInRu = [...ruKeys].filter(k => !enKeys.has(k));
const extraInUz = [...uzKeys].filter(k => !enKeys.has(k));

console.log('=== TRANSLATION KEY CONSISTENCY CHECK ===\n');
console.log(`Total keys in en.json: ${enKeys.size}`);
console.log(`Total keys in ru.json: ${ruKeys.size}`);
console.log(`Total keys in uz.json: ${uzKeys.size}\n`);

if (missingInRu.length > 0) {
  console.log(`❌ Keys missing in ru.json (${missingInRu.length}):`);
  missingInRu.forEach(k => console.log(`  - ${k}`));
} else {
  console.log('✅ ru.json has all keys from en.json');
}

if (missingInUz.length > 0) {
  console.log(`\n❌ Keys missing in uz.json (${missingInUz.length}):`);
  missingInUz.forEach(k => console.log(`  - ${k}`));
} else {
  console.log('\n✅ uz.json has all keys from en.json');
}

if (extraInRu.length > 0) {
  console.log(`\n⚠️  Extra keys in ru.json not in en.json (${extraInRu.length}):`);
  extraInRu.forEach(k => console.log(`  - ${k}`));
}

if (extraInUz.length > 0) {
  console.log(`\n⚠️  Extra keys in uz.json not in en.json (${extraInUz.length}):`);
  extraInUz.forEach(k => console.log(`  - ${k}`));
}

if (missingInRu.length === 0 && missingInUz.length === 0 && extraInRu.length === 0 && extraInUz.length === 0) {
  console.log('\n✅ All translation files are perfectly synchronized!');
}
