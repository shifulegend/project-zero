import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// TypeScript in <script lang="ts"> blocks is stripped by Vite's own preprocessor; types are checked
// separately by svelte-check (npm run check).
export default {
  preprocess: vitePreprocess(),
};
