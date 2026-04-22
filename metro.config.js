const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");

const config = getDefaultConfig(__dirname);

// Fix: "Cannot use 'import.meta' outside a module" on web (e.g. Zustand ESM uses import.meta)
config.resolver.unstable_enablePackageExports = false;

// Support 3D model assets (.glb, .gltf)
config.resolver.assetExts = [
  ...(config.resolver.assetExts || []),
  'glb',
  'gltf',
];

module.exports = withNativeWind(config, { input: "./global.css" });