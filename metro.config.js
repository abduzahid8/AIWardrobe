const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");
// Stamp the source Metro actually serves, including uncommitted edits.
const { execFileSync } = require('node:child_process');
const sha = execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], {cwd: __dirname, encoding: 'utf8'}).trim();
const dirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: __dirname, encoding: 'utf8'}).trim();
process.env.EXPO_PUBLIC_CLIENT_REVISION = `${sha}${dirty ? '+dirty' : ''}`;

const config = getDefaultConfig(__dirname);

// Zustand v5 / i18n ship proper ESM via package exports — enabling this lets
// Metro tree-shake them (smaller Hermes bytecode, faster startup).
// `unstable_transformImportMeta: true` in babel.config.js already handles the
// web `import.meta` case, so the old `false` workaround is no longer needed.
config.resolver.unstable_enablePackageExports = true;

// Support 3D model assets (.glb, .gltf)
config.resolver.assetExts = [
  ...(config.resolver.assetExts || []),
  'glb',
  'gltf',
];

module.exports = withNativeWind(config, { input: "./global.css" });
