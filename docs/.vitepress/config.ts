import { defineConfig } from 'vitepress';
import sidebars from './sidebar';

export default defineConfig({
  base: '/djtu-wiki/',
  title: 'djtu-wiki',
  description: 'Djtu生存指南科学讲义',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { 
        text: 'Guide', 
        items: [
          { text: '立志篇', link: '/guide/aspire' },
          { text: '方向篇', link: '/guide/direction' },
          { text: '生存篇', link: '/guide/being' }
        ]
      },
      {
        text: 'Programming Languages',
        items: [
          { text: 'C & C++', link: '/language/cpp/' },
          { text: 'Python', link: '/language/python/' },
          { text: 'Rust', link: '/language/rust/' }
        ]
      },
      {
        text: 'Technology Stack',
        items: [
          { text: '算法', link: '/jobs/algorithm/1' },
          { text: '人工智能', link: '/jobs/ai/hello' },
          { text: 'Web 开发', link: '/jobs/Web/hello' },
          { text: '嵌入式开发', link: '/jobs/embedded/hello' },
          { text: 'typst和latex', link: '/jobs/Typst/latex' }
        ]
      },
      { 
        text: 'Course', 
        items: [
          { text: '泛计算机类资源搜集', link: '/course' },
        ]
      }
    ],
    // 将 sidebar 放在 themeConfig 内
    sidebar: sidebars
  }
});
