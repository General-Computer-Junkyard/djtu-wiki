// 导入自动生成的侧边栏配置
import sidebar from './sidebar'

export default {
  title: 'DJTU-wiki', // 网站标题
  description: 'Djtu生存指南科学讲义', // 网站描述
  themeConfig: {
    nav: [ // 手动配置的顶部导航栏
      { text: 'Home', link: '/' },
      {
        text: 'Programming Languages',
        items: [
          { text: 'C & C++', link: '/cpp/' },
          { text: 'Python', link: '/python/' },
          { text: 'Rust', link: '/language/rust/' }
        ]
      },
      { 
        text: 'Technology Stack', 
        items: [
          { text: '算法', link: '/jobs/algorithm/'},
          { text: '人工智能', link: '/jobs/ai/'},
          { text: 'Web 开发', link: '/jobs/web/'},
          { text: '嵌入式开发', link: '/jobs/embedded/' }
        ]
      },
      { text: 'Config', link: '/config/' },
      { text: 'Course Guide', link: '/course/' }
    ],
    sidebar: sidebar // 使用自动生成的侧边栏配置
  }
}
