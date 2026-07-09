/* Minimal ESLint config for the Vite + React app.
 *
 * The project shipped a `lint` script and the eslint-plugin-react* devDeps but
 * no config file, so `npm run lint` errored out ("couldn't find a configuration
 * file"). This restores a working config. Classic React runtime (files import
 * React), so plugin:react/recommended's jsx-uses-react keeps React "used".
 *
 * NOTE: repo-wide warning cleanup is a separate task — this only makes lint run.
 */
module.exports = {
  root: true,
  env: { browser: true, es2021: true, node: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
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
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    'no-empty': ['warn', { allowEmptyCatch: true }],
  },
}
