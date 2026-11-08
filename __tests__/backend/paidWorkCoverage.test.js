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
  // Admission may be traced (`await trace.wait('admission', () => enforceRateLimit(...))`).
  // Unwrap awaits, arrow bodies and wait() wrappers to find the actual call;
  // the authorization guarantees are asserted on the call itself.
  function admissionCall(root) {
    let current = root;
    while (current) {
      if (ts.isAwaitExpression(current)) {
        current = current.expression;
      } else if (ts.isArrowFunction(current)) {
        const body = current.body;
        if (ts.isBlock(body)) {
          const ret = body.statements.find(ts.isReturnStatement);
          current = ret?.expression;
        } else {
          current = body;
        }
      } else if (ts.isCallExpression(current)) {
        const callee = current.expression.getText(ast);
        if (callee === 'enforceRateLimit') return current;
        if (callee.endsWith('.wait')) {
          current = current.arguments[current.arguments.length - 1];
        } else {
          return null;
        }
      } else {
        return null;
      }
    }
    return null;
  }
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.initializer) {
      const call = admissionCall(node.initializer);
      if (call) {
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
