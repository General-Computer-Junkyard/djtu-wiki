import mathjax3 from "markdown-it-mathjax3";

const customElements = [
    "math",
    "maction",
    "maligngroup",
    "malignmark",
    "menclose",
    "merror",
    "mfenced",
    "mfrac",
    "mi",
    "mlongdiv",
    "mmultiscripts",
    "mn",
    "mo",
    "mover",
    "mpadded",
    "mphantom",
    "mroot",
    "mrow",
    "ms",
    "mscarries",
    "mscarry",
    "mscarries",
    "msgroup",
    "mstack",
    "mlongdiv",
    "msline",
    "mstack",
    "mspace",
    "msqrt",
    "msrow",
    "mstack",
    "mstack",
    "mstyle",
    "msub",
    "msup",
    "msubsup",
    "mtable",
    "mtd",
    "mtext",
    "mtr",
    "munder",
    "munderover",
    "semantics",
    "math",
    "mi",
    "mn",
    "mo",
    "ms",
    "mspace",
    "mtext",
    "menclose",
    "merror",
    "mfenced",
    "mfrac",
    "mpadded",
    "mphantom",
    "mroot",
    "mrow",
    "msqrt",
    "mstyle",
    "mmultiscripts",
    "mover",
    "mprescripts",
    "msub",
    "msubsup",
    "msup",
    "munder",
    "munderover",
    "none",
    "maligngroup",
    "malignmark",
    "mtable",
    "mtd",
    "mtr",
    "mlongdiv",
    "mscarries",
    "mscarry",
    "msgroup",
    "msline",
    "msrow",
    "mstack",
    "maction",
    "semantics",
    "annotation",
    "annotation-xml",
    "mjx-container",
    "mjx-assistive-mml",
];

import { defineConfig } from "vitepress";
import sidebars from "./sidebar";

export default defineConfig({
  base: "/djtu-wiki/",
  title: "djtu-wiki",
  description: "Djtu生存指南科学讲义",
  head: [
    [
      "link",
      {
        rel: "stylesheet",
        href: "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css",
      },
    ],
  ],
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      {
        text: "Guide",
        items: [
          { text: "立志篇", link: "/guide/aspire" },
          { text: "方向篇", link: "/guide/direction" },
          { text: "生存篇", link: "/guide/being" },
        ],
      },
      {
        text: "Programming Languages",
        items: [
          { text: "C & C++", link: "/language/cpp/" },
          { text: "Python", link: "/language/python/" },
          { text: "Rust", link: "/language/rust/" },
        ],
      },
      {
        text: "Technology Stack",
        items: [
          { text: "算法", link: "/jobs/algorithm/1" },
          { text: "人工智能", link: "/jobs/ai/hello" },
          { text: "Web 开发", link: "/jobs/Web/hello" },
          { text: "嵌入式开发", link: "/jobs/embedded/hello" },
          { text: "typst和latex", link: "/jobs/Typst/latex" },
        ],
      },
      {
        text: "Course",
        items: [
          { text: "泛计算机类资源搜集", link: "/course" },
        ],
      },
    ],
    sidebar: sidebars,
  },
  markdown: {
    config: (md) => {
      md.use(mathjax3);
    },
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },
  vite: {
    ssr: {
      noExternal: ["vue"],
    },
  },
});
