/** @jest-environment node */
// Server safety contract: every currently paid Edge entry uses authenticated
// shared admission and returns its denial; clients cannot choose meterPaidWork.
const fs = require('node:fs'),
  path = require('node:path'),
  ts = require('typescript');
const paid = [
  'ai-chat',
  'ai-process',
  'analyze-outfit',
  'create-trip-plan',
  'generate-outfits',
  'try-on',
];
test.each(paid)('%s cannot opt out of shared paid-work admission', (endpoint) => {
  const source = fs.readFileSync(
    path.resolve(__dirname, `../../backend/supabase/functions/${endpoint}/index.ts`),
    'utf8'
  );
  const ast = ts.createSourceFile(
    endpoint + '.ts',
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS
  );
  let binding,
    matched = false,
    returnsDenied = false;
  function visit(node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.initializer &&
      ts.isAwaitExpression(node.initializer)
    ) {
      const call = node.initializer.expression;
      if (ts.isCallExpression(call) && call.expression.getText(ast) === 'enforceRateLimit') {
        const options = call.arguments[1];
        expect(ts.isObjectLiteralExpression(options)).toBe(true);
        const props = new Map(
          options.properties
            .filter(ts.isPropertyAssignment)
            .map((p) => [p.name.getText(ast), p.initializer])
        );
        expect(props.get('feature')?.text).toBe(endpoint);
        const mode = props.get('meterPaidWork');
        expect(!mode || mode.kind === ts.SyntaxKind.TrueKeyword).toBe(true);
        binding = node.name.getText(ast);
        matched = true;
      }
    }
    if (ts.isIfStatement(node) && binding && node.expression.getText(ast) === binding) {
      const statement = ts.isBlock(node.thenStatement)
        ? node.thenStatement.statements[0]
        : node.thenStatement;
      if (ts.isReturnStatement(statement) && statement.expression?.getText(ast) === binding)
        returnsDenied = true;
    }
    ts.forEachChild(node, visit);
  }
  visit(ast);
  expect(matched).toBe(true);
  expect(returnsDenied).toBe(true);
  expect(source).toMatch(
    /import\s*\{\s*enforceRateLimit\s*\}\s*from\s*['"]\.\.\/_shared\/rateLimit\.ts['"]/
  );
});
