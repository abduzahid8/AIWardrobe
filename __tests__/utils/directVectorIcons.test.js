const { transformSync } = require('@babel/core');
const plugin = require('../../scripts/babel/direct-vector-icons.cjs');
const compile = source => transformSync(source, { configFile: false, babelrc: false, plugins: [plugin] }).code;
test('font families resolve directly with aliases intact', () => {
  const output = compile("import { Ionicons as Icon, MaterialCommunityIcons } from '@expo/vector-icons';");
  expect(output).toContain('import Icon from "@expo/vector-icons/Ionicons"');
  expect(output).toContain('import MaterialCommunityIcons from "@expo/vector-icons/MaterialCommunityIcons"');
  expect(output).not.toMatch(/from ['"]@expo\/vector-icons['"]/);
});
test('helpers, namespaces and side effects retain their behavior', () => {
  expect(compile("import * as Icons from '@expo/vector-icons';")).toContain("import * as Icons from '@expo/vector-icons'");
  const output = compile("import { createIconSet, Ionicons } from '@expo/vector-icons';");
  expect(output).toContain('createIconSet');
  expect(output).toMatch(/from ['"]@expo\/vector-icons['"]/);
  expect(output).toContain('from "@expo/vector-icons/Ionicons"');
  expect(compile("import '@expo/vector-icons';")).toContain("import '@expo/vector-icons'");
});
