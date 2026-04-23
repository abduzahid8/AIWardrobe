module.exports = {
    root: true,
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:react/recommended',
        'plugin:react-hooks/recommended',
    ],
    parser: '@typescript-eslint/parser',
    parserOptions: {
        ecmaVersion: 2021,
        sourceType: 'module',
        ecmaFeatures: {
            jsx: true,
        },
    },
    plugins: ['@typescript-eslint', 'react', 'react-hooks'],
    env: {
        browser: true,
        node: true,
        es2021: true,
    },
    settings: {
        react: {
            version: 'detect',
        },
    },
    rules: {
        // --- TypeScript discipline -----------------------------
        // `any` is a ratcheting rule: warn today, error once we are clean.
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/no-unused-vars': [
            'warn',
            { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
        ],
        '@typescript-eslint/explicit-function-return-type': 'off',
        '@typescript-eslint/explicit-module-boundary-types': 'off',
        '@typescript-eslint/no-empty-function': 'warn',
        '@typescript-eslint/ban-ts-comment': 'warn',
        '@typescript-eslint/no-var-requires': 'warn',

        // --- React --------------------------------------------
        'react/react-in-jsx-scope': 'off',
        'react/prop-types': 'off',
        'react/display-name': 'off',
        'react/no-unescaped-entities': 'warn',
        'react/jsx-no-comment-textnodes': 'warn',

        // --- React Hooks --------------------------------------
        'react-hooks/rules-of-hooks': 'error',
        'react-hooks/exhaustive-deps': 'warn',

        // --- Logging discipline -------------------------------
        // Use src/utils/logger.ts instead of console.*. The only
        // exceptions are bootstrap files that run before the logger
        // module is importable (see overrides below).
        'no-console': ['error', { allow: ['warn', 'error'] }],

        // --- General ------------------------------------------
        'no-debugger': 'error',
        'prefer-const': 'warn',
        'no-var': 'error',
        'eqeqeq': ['warn', 'always', { null: 'ignore' }],
        'no-empty': ['warn', { allowEmptyCatch: true }],
        'no-useless-escape': 'warn',
        'no-undef': 'warn',

        // --- Guardrails we actually care about ----------------
        // Force users toward the logger abstraction.
        'no-restricted-imports': [
            'error',
            {
                paths: [
                    {
                        name: '@google/generative-ai',
                        message:
                            'Do not call AI providers from the client. Use supabase.functions.invoke("ai-process").',
                    },
                    {
                        name: '@huggingface/inference',
                        message:
                            'Do not call AI providers from the client. Use supabase.functions.invoke("ai-process").',
                    },
                    {
                        name: 'replicate',
                        message:
                            'Do not call AI providers from the client. Use supabase.functions.invoke("ai-process").',
                    },
                ],
                patterns: [
                    {
                        group: ['**/api/**', '../api/*', '../../api/*'],
                        message:
                            'The api/ Express tree is deprecated (see docs/ARCHITECTURE.md ADR-001). Call Supabase Edge Functions instead.',
                    },
                ],
            },
        ],
    },
    overrides: [
        {
            // API files are JavaScript / node
            files: ['api/**/*.js', 'scripts/**/*.js', 'scripts/**/*.ts'],
            env: { node: true, browser: false },
            rules: {
                '@typescript-eslint/no-var-requires': 'off',
                '@typescript-eslint/no-require-imports': 'off',
                'no-console': 'off',
                'no-restricted-imports': 'off',
            },
        },
        {
            // Test files
            files: [
                '**/*.test.ts',
                '**/*.test.tsx',
                '**/*.spec.ts',
                '**/*.spec.tsx',
                '**/__tests__/**',
                'jest.setup.js',
            ],
            env: { jest: true },
            rules: {
                '@typescript-eslint/no-explicit-any': 'off',
                'no-console': 'off',
            },
        },
        {
            // The logger itself has to call console.*
            files: ['src/utils/logger.ts'],
            rules: { 'no-console': 'off' },
        },
        {
            // Supabase Edge Functions run on Deno, not Node.
            files: ['supabase/functions/**'],
            rules: {
                'no-console': 'off',
                'no-undef': 'off',
                '@typescript-eslint/no-explicit-any': 'off',
            },
        },
    ],
    ignorePatterns: [
        'node_modules/',
        'dist/',
        'dist-*/',
        '.expo/',
        'babel.config.js',
        'metro.config.js',
        'tailwind.config.js',
        'api/node_modules/',
        'alicevision-service/',
        'supabase/.temp/',
        'coverage/',
    ],
};
