import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
//import './style.css'

import './tailwind.css'


export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout)
  },
  enhanceApp({ app }) {
    // 注册全局组件如果需要
  }
}