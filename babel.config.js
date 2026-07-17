module.exports = function (api) {
  api.cache(true);

  const plugins = [
    [
      "module-resolver",
      {
        root: ["."],
        alias: {
          "@": "./src",
          "@components": "./components",
          "@screens": "./screens",
          "@store": "./store",
          "@hooks": "./hooks",
          "@navigation": "./navigation",
          "@constants": "./constants",
          "@lib": "./lib",
          "@assets": "./assets",
        },
      },
    ],
    "react-native-reanimated/plugin",
  ];

  // Strip console.* in production to prevent sensitive data leaks
  // and improve JS thread performance. Also strips debug/info/perf logs
  // in dev to keep the Metro console clean.
  const isProduction = process.env.NODE_ENV === 'production' || process.env.BABEL_ENV === 'production';
  if (isProduction) {
    plugins.push(['transform-remove-console', { exclude: ['error', 'warn'] }]);
  }



  return {
    presets: [
      [
        "babel-preset-expo",
        {
          jsxImportSource: "nativewind",
          unstable_transformImportMeta: true,
        },
      ],
      "nativewind/babel",
    ],
    plugins,
  };
};