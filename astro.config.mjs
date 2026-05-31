// @ts-check
import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeCodeDetails from "./src/lib/rehype-code-details.mjs";

export default defineConfig({
  site: "https://haoyang9804.github.io",
  output: "static",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeCodeDetails],
    syntaxHighlight: "shiki",
    shikiConfig: {
      theme: "github-dark"
    }
  }
});
