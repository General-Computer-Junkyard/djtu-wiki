---
outline: deep
---

<script setup>
import { defineClientComponent } from 'vitepress'
const TetrisGame = defineClientComponent(() => import('./TetrisGame.vue'))
</script>
# web

## web内容尚未更新,但是你来都来了,玩一局俄罗斯方块再回去吧() 


<div class="game-container">
  <div class="game-wrapper">
    来玩个小游戏吧,这里的配色是三月七的配色,我是三月七单推人!
    <ClientOnly>
      <TetrisGame />
    </ClientOnly>
  </div>
</div>

<style>
.game-container {
  display: flex;
  justify-content: center;
  width: 100%;
  margin: 2rem 0;
}

.game-wrapper {
  max-width: 500px;
  width: 100%;
  text-align: center;
}
</style>