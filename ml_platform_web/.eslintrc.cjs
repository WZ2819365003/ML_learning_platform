/* Minimal ESLint config for the Vite + React app.
 *
 * The project shipped a `lint` script and the eslint-plugin-react* devDeps but
 * no config file, so `npm run lint` errored out ("couldn't find a configuration
 * file"). Vite uses the automatic JSX runtime, including files that do not need
 * an explicit React import.
 */
module.exports = {
  root: true,
  env: { browser: true, es2021: true, node: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
  settings: { react: { version: 'detect' } },
  plugins: ['react-refresh'],
  ignorePatterns: [
    'dist', 'node_modules', 'playwright-report', 'test-results', '.eslintrc.cjs',
  ],
  rules: {
    'react/prop-types': 'off',
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^(React|_)' }],
    'no-empty': ['warn', { allowEmptyCatch: true }],
  },
}
