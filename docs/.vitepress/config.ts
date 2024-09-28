import { defineConfig } from 'vitepress';
import sidebars from './sidebar';

export default defineConfig({
  base:'/DJTU-wiki/',
  title: 'DJTU-wiki',
  description: 'Djtu生存指南科学讲义',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
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
          { text: '算法', link: '/jobs/algorithm/' },
          { text: '人工智能', link: '/jobs/ai/' },
          { text: 'Web 开发', link: '/jobs/web/' },
          { text: '嵌入式开发', link: '/jobs/embedded/' }
        ]
      },
      { text: 'Config', link: '/config/' },
      { text: 'Course Guide', link: '/course/' }
    ],

    // 侧边栏配置
    sidebar: sidebars
  }
});
