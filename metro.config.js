const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");

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