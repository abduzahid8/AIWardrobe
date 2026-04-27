const fs = require('fs');
const path = require('path');

// Get all TSX files in screens and components
const screensDir = path.join(__dirname, 'screens');
const componentsDir = path.join(__dirname, 'components');

function getTsxFiles(dir) {
  const files = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const item of items) {
    const fullPath = path.join(dir, item.name);
    if (item.isDirectory()) {
      files.push(...getTsxFiles(fullPath));
    } else if (item.name.endsWith('.tsx')) {
      files.push(fullPath);
    }
  }
  return files;
}

const screenFiles = getTsxFiles(screensDir);
const componentFiles = getTsxFiles(componentsDir);
const allFiles = [...screenFiles, ...componentFiles];

console.log('=== COMPONENTS WITHOUT useTranslation ===\n');
console.log(`Total TSX files found: ${allFiles.length}\n`);

const filesWithoutI18n = [];

for (const file of allFiles) {
  const content = fs.readFileSync(file, 'utf8');
  const hasUseTranslation = content.includes('useTranslation');
  const hasTFunction = content.includes('t(') || content.includes('t(');
  const hasTransHook = content.includes('useTranslation');
  
  // Check if file has user-facing text (Text components with strings)
  const hasTextComponents = /<Text[^>]*>[^<]*[a-zA-Z]+[^<]*<\/Text>/.test(content);
  const hasStringLiterals = /["']([A-Z][a-zA-Z\s]{5,})["']/.test(content);
  
  if (!hasUseTranslation && (hasTextComponents || hasStringLiterals)) {
    filesWithoutI18n.push({
      file: path.relative(__dirname, file),
      hasTextComponents,
      hasStringLiterals
    });
  }
}

if (filesWithoutI18n.length === 0) {
  console.log('✅ All components with user-facing text are using useTranslation');
} else {
  console.log(`❌ Found ${filesWithoutI18n.length} files without useTranslation but with user-facing text:\n`);
  filesWithoutI18n.forEach(({ file, hasTextComponents, hasStringLiterals }) => {
    console.log(`  - ${file}`);
    if (hasTextComponents) console.log('    (has Text components)');
    if (hasStringLiterals) console.log('    (has string literals)');
  });
}

// Also check for files that import useTranslation but might not be using it properly
console.log('\n=== FILES WITH useTranslation IMPORT ===\n');
const filesWithImport = allFiles.filter(file => {
  const content = fs.readFileSync(file, 'utf8');
  return content.includes('useTranslation');
});

console.log(`Files importing useTranslation: ${filesWithImport.length}/${allFiles.length}\n`);

// Check for hardcoded English strings in files that DO use useTranslation
console.log('=== POTENTIAL HARDCODED STRINGS IN i18n FILES ===\n');
const suspiciousFiles = [];

for (const file of filesWithImport) {
  const content = fs.readFileSync(file, 'utf8');
  // Look for common hardcoded patterns
  const patterns = [
    /<Text[^>]*>["']([A-Z][a-zA-Z\s]{5,})["']<\/Text>/g,
    /title=["']([A-Z][a-zA-Z\s]{5,})["']/g,
    /placeholder=["']([A-Z][a-zA-Z\s]{5,})["']/g,
  ];
  
  for (const pattern of patterns) {
    const matches = content.match(pattern);
    if (matches) {
      suspiciousFiles.push({
        file: path.relative(__dirname, file),
        matches: matches.slice(0, 3) // Show first 3 matches
      });
      break;
    }
  }
}

if (suspiciousFiles.length === 0) {
  console.log('✅ No obvious hardcoded strings found in i18n files');
} else {
  console.log(`⚠️  Found ${suspiciousFiles.length} files with potential hardcoded strings:\n`);
  suspiciousFiles.forEach(({ file, matches }) => {
    console.log(`  - ${file}`);
    matches.forEach(m => console.log(`    ${m}`));
  });
}
