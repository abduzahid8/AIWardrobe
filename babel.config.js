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
  if (process.env.NODE_ENV === 'production' || process.env.BABEL_ENV === 'production') {
    plugins.push('transform-remove-console');
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