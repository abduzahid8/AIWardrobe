const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");
// Stamp the source Metro actually serves, including uncommitted edits, while
// remaining resilient in EAS Build workers where `.easignore` excludes `.git`.
const { execFileSync } = require('node:child_process');

/** Resolve a short client revision identifier without crashing when `.git` is excluded. */
function resolveClientRevision() {
  const preset = (process.env.EXPO_PUBLIC_CLIENT_REVISION || '').trim();
  if (preset) return preset;
  try {
    const sha = execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], {
      cwd: __dirname,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    if (sha) {
      const dirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {
        cwd: __dirname,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
      }).trim();
      return `${sha}${dirty ? '+dirty' : ''}`;
    }
  } catch {
    // EAS cloud archives omit `.git` via `.easignore`.
  }
  return (process.env.EAS_BUILD_GIT_COMMIT_HASH || '').trim().slice(0, 12) || 'unstamped';
}

process.env.EXPO_PUBLIC_CLIENT_REVISION = resolveClientRevision();

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
