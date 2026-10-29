const { transformSync } = require('@babel/core');
const plugin = require('../../scripts/babel/direct-vector-icons.cjs');
const compile = source => transformSync(source, { configFile: false, babelrc: false, plugins: [plugin] }).code;

test('icon families compile to direct entry points with aliases preserved', () => {
  const output = compile("import { Ionicons as Icon, MaterialCommunityIcons } from '@expo/vector-icons';");
  expect(output).toContain('import Icon from "@expo/vector-icons/Ionicons"');
  expect(output).toContain('import MaterialCommunityIcons from "@expo/vector-icons/MaterialCommunityIcons"');
  expect(output).not.toContain('from "@expo/vector-icons"');
});
test('namespace, side-effect and helper imports retain their original behavior', () => {
  expect(compile("import * as Icons from '@expo/vector-icons';")).toContain("import * as Icons from '@expo/vector-icons'");
  const output = compile("import { createIconSet, Ionicons } from '@expo/vector-icons';");
  expect(output).toContain('createIconSet');
  expect(output).toMatch(/from ['"]@expo\/vector-icons['"]/);
  expect(output).toContain('from "@expo/vector-icons/Ionicons"');
  expect(compile("import '@expo/vector-icons';")).toContain("import '@expo/vector-icons'");
});
