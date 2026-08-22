const fs = require('fs');
const path = require('path');

const screensDir = './screens';

function findHardcodedText(dir) {
  const issues = [];
  const files = fs.readdirSync(dir, { withFileTypes: true });

  for (const file of files) {
    const fullPath = path.join(dir, file.name);
    if (file.isDirectory()) {
      if (!['node_modules', '.git'].includes(file.name)) {
        issues.push(...findHardcodedText(fullPath));
      }
    } else if (file.name.match(/\.(ts|tsx|js|jsx)$/)) {
      const content = fs.readFileSync(fullPath, 'utf8');
      const lines = content.split('\n');

      lines.forEach((line, index) => {
        // Find Text components with hardcoded English text (not using t())
        // Pattern: <Text>Some English text</Text> or <Text style={...}>Some English text</Text>
        const textComponentMatches = line.matchAll(/<Text[^>]*>([^<]+)<\/Text>/g);
        
        for (const match of textComponentMatches) {
          const text = match[1].trim();
          // Skip if it's empty, just numbers, or already using t()
          if (text && 
              text.length > 2 && 
              !text.startsWith('{') && 
              !text.startsWith('$') &&
              !text.match(/^\d+$/) &&
              !text.includes('t(') &&
              !text.includes('formatDate') &&
              !text.includes('TIER_CARDS') &&
              !text.includes('SUBSCRIPTION_PRICING') &&
              // Check if it looks like English (has letters and spaces)
              /[a-zA-Z]{3,}/.test(text) &&
              // Skip common single words that might be OK
              !['AI', 'Pro', 'Max', 'Free', 'LIVE', 'HRS', 'MIN', 'SEC', 'SKU', 'ID', 'URL', 'OK', 'X'].includes(text.toUpperCase())) {
            issues.push({
              file: fullPath,
              line: index + 1,
              text: text
            });
          }
        }

        // Find hardcoded strings in JavaScript (not using t())
        // Pattern: "Some English text" or 'Some English text' that's not part of t()
        const stringMatches = line.matchAll(/["']([A-Z][a-z]+ [a-z]+(?: [a-z]+)*)["']/g);
        
        for (const match of stringMatches) {
          const text = match[1];
          // Skip if it's in a t() call or already processed
          if (text && 
              !line.includes('t(') &&
              !line.includes('formatDate') &&
              !line.includes('console.') &&
              !line.includes('logger.') &&
              !line.includes('debug(') &&
              !line.includes('error(') &&
              !line.includes('warn(') &&
              !line.includes('info(')) {
            issues.push({
              file: fullPath,
              line: index + 1,
              text: text,
              context: line.trim()
            });
          }
        }
      });
    }
  }
  return issues;
}

const issues = findHardcodedText(screensDir);

console.log('=== HARDCODED TEXT FOUND ===');
console.log(`Count: ${issues.length}\n`);

issues.forEach(issue => {
  console.log(`${issue.file}:${issue.line}`);
  console.log(`  Text: "${issue.text}"`);
  if (issue.context) {
    console.log(`  Context: ${issue.context}`);
  }
  console.log();
});
