<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'

// 定义方块形状
const TETROMINOES = {
  I: [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ],
  O: [
    [1, 1],
    [1, 1],
  ],
  T: [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  L: [
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
  ],
  J: [
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  S: [
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
  ],
  Z: [
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
  ],
}

const COLORS = {
  I: '#21839E',
  O: '#7CB5B4',
  T: '#7F6FC1',
  L: '#A8AECB',
  J: '#48AFE8',
  S: '#F2ABF7',
  Z: '#BB7593',
}

// 游戏状态
const gameBoard = ref(Array(20).fill().map(() => Array(10).fill(0)))
const currentPiece = ref(null)
const currentPosition = ref({ x: 0, y: 0 })
const score = ref(0)
const gameOver = ref(false)
const isPaused = ref(false)

// 创建新方块
const createNewPiece = () => {
  const pieces = Object.keys(TETROMINOES)
  const randomPiece = pieces[Math.floor(Math.random() * pieces.length)]
  currentPiece.value = {
    shape: TETROMINOES[randomPiece],
    type: randomPiece,
  }
  currentPosition.value = { x: 3, y: 0 }
}

// 检查碰撞
const checkCollision = (piece, position) => {
  for (let y = 0; y < piece.shape.length; y++) {
    for (let x = 0; x < piece.shape[y].length; x++) {
      if (piece.shape[y][x]) {
        const newX = position.x + x
        const newY = position.y + y
        
        if (
          newX < 0 || 
          newX >= 10 || 
          newY >= 20 ||
          (newY >= 0 && gameBoard.value[newY][newX])
        ) {
          return true
        }
      }
    }
  }
  return false
}

// 合并方块到游戏板
const mergePiece = () => {
  const newBoard = [...gameBoard.value]
  for (let y = 0; y < currentPiece.value.shape.length; y++) {
    for (let x = 0; x < currentPiece.value.shape[y].length; x++) {
      if (currentPiece.value.shape[y][x]) {
        const boardY = currentPosition.value.y + y
        if (boardY < 0) {
          gameOver.value = true
          return
        }
        newBoard[boardY][currentPosition.value.x + x] = currentPiece.value.type
      }
    }
  }
  gameBoard.value = newBoard
}

// 清除完整的行
const clearLines = () => {
  const newBoard = gameBoard.value.filter(row => !row.every(cell => cell))
  const clearedLines = gameBoard.value.length - newBoard.length
  score.value += clearedLines * 100
  
  while (newBoard.length < 20) {
    newBoard.unshift(Array(10).fill(0))
  }
  
  gameBoard.value = newBoard
}

// 移动方块
const movePiece = (dx, dy) => {
  const newPosition = { 
    x: currentPosition.value.x + dx, 
    y: currentPosition.value.y + dy 
  }
  
  if (!checkCollision(currentPiece.value, newPosition)) {
    currentPosition.value = newPosition
    return true
  }
  
  if (dy > 0) {
    mergePiece()
    clearLines()
    createNewPiece()
  }
  return false
}

// 旋转方块
const rotatePiece = () => {
  if (!currentPiece.value) return
  
  const newShape = currentPiece.value.shape[0].map((_, i) =>
    currentPiece.value.shape.map(row => row[i]).reverse()
  )
  
  const rotatedPiece = { 
    ...currentPiece.value, 
    shape: newShape 
  }
  
  if (!checkCollision(rotatedPiece, currentPosition.value)) {
    currentPiece.value = rotatedPiece
  }
}

// 处理键盘事件
const handleKeyPress = (e) => {
  if (gameOver.value || isPaused.value) return
  
  switch (e.key.toLowerCase()) {
    case 'a':
      movePiece(-1, 0)
      break
    case 'd':
      movePiece(1, 0)
      break
    case 's':
      movePiece(0, 1)
      break
    case 'w':
      rotatePiece()
      break
    case ' ':
      while (movePiece(0, 1)) {}
      break
    case 'p':
      isPaused.value = !isPaused.value
      break
  }
}

// 获取方块颜色
const getCellColor = (row, col) => {
  // 检查当前移动的方块
  if (
    currentPiece.value &&
    row >= currentPosition.value.y &&
    row < currentPosition.value.y + currentPiece.value.shape.length &&
    col >= currentPosition.value.x &&
    col < currentPosition.value.x + currentPiece.value.shape[0].length &&
    currentPiece.value.shape[row - currentPosition.value.y][col - currentPosition.value.x]
  ) {
    return COLORS[currentPiece.value.type]
  }
  // 检查已固定的方块
  return gameBoard.value[row][col] ? COLORS[gameBoard.value[row][col]] : '#1f2937'
}

// 重启游戏
const restartGame = () => {
  gameBoard.value = Array(20).fill().map(() => Array(10).fill(0))
  currentPiece.value = null
  score.value = 0
  gameOver.value = false
  isPaused.value = false
  createNewPiece()
}

// 游戏循环
let gameLoop
onMounted(() => {
  window.addEventListener('keydown', handleKeyPress)
  createNewPiece()
  
  gameLoop = setInterval(() => {
    if (!isPaused.value && !gameOver.value && currentPiece.value) {
      movePiece(0, 1)
    }
  }, 1000)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyPress)
  clearInterval(gameLoop)
})
</script>

<template>
  <div class="tetris-game">
    <div class="mb-4">
      <div class="text-xl font-bold">得分: {{ score }}</div>
      <div v-if="gameOver" class="text-red-500 font-bold">游戏结束!</div>
      <div v-if="isPaused" class="text-blue-500 font-bold">游戏暂停</div>
    </div>
    
    <div class="tetris-board">
      <div v-for="(row, y) in gameBoard" :key="y" class="flex">
        <div
          v-for="(cell, x) in row"
          :key="x"
          class="tetris-cell"
          :style="{ backgroundColor: getCellColor(y, x) }"
        />
      </div>
    </div>

    <div class="mt-4 space-x-4">
      <button
        @click="restartGame"
        class="tetris-button tetris-button-primary"
      >
        重新开始
      </button>
      <button
        @click="isPaused = !isPaused"
        class="tetris-button tetris-button-secondary"
      >
        {{ isPaused ? '继续' : '暂停' }}
      </button>
    </div>

    <div class="tetris-controls">
      <p>控制方式:</p>
      <ul class="list-disc list-inside">
        <li>A/D: 左右移动方块</li>
        <li>W: 旋转方块</li>
        <li>S: 加速下落</li>
        <li>空格键: 直接落地</li>
        <li>P键: 暂停/继续</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.tetris-game {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  border-radius: 0.5rem;
  margin: 1rem 0;
}

.tetris-board {
  border: 4px solid #1f2937;
  background-color: #111827;
  display: inline-block;
}

.tetris-cell {
  width: 24px;
  height: 24px;
  border: 1px solid #374151;
  display: inline-block;
}

.tetris-button {
  padding: 0.5rem 1rem;
  color: white;
  border-radius: 0.25rem;
  margin: 0 0.5rem;
  cursor: pointer;
}

.tetris-button-primary {
  background-color: #3b82f6;
}

.tetris-button-primary:hover {
  background-color: #2563eb;
}

.tetris-button-secondary {
  background-color: #10b981;
}

.tetris-button-secondary:hover {
  background-color: #059669;
}

.tetris-controls {
  margin-top: 1rem;
  font-size: 0.875rem;
  color: #4b5563;
}

.flex {
  display: flex;
}

.mb-4 {
  margin-bottom: 1rem;
}

.mt-4 {
  margin-top: 1rem;
}

.space-x-4 > * + * {
  margin-left: 1rem;
}

.text-xl {
  font-size: 1.25rem;
  line-height: 1.75rem;
}

.font-bold {
  font-weight: 700;
}

.text-red-500 {
  color: #ef4444;
}

.text-blue-500 {
  color: #3b82f6;
}

.list-disc {
  list-style-type: disc;
  padding-left: 1.5rem;
}
</style>