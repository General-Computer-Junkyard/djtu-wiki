<script setup>
import { ref, onMounted, watch } from 'vue'
import * as tf from '@tensorflow/tfjs'

const canvasRef = ref(null)
const networkCanvasRef = ref(null)
const resultCanvasRef = ref(null)
const isDrawing = ref(false)
const ctx = ref(null)
const predictedNumber = ref(null)
const networkLayers = ref([
  { name: 'Input', neurons: 20 },    // 减少显示数量
  { name: 'Conv1', neurons: 12 },
  { name: 'Pool1', neurons: 8 },
  { name: 'Conv2', neurons: 12 },
  { name: 'Pool2', neurons: 8 },
  { name: 'Dense', neurons: 10 },
  { name: 'Output', neurons: 10 }
])
const activations = ref(new Array(networkLayers.value.length).fill(0))

const model = ref(null)

const isStarted = ref(false)  // 添加开始状态控制
const isPredicting = ref(false)  // 添加预测状态控制

// 加载模型
const loadModel = async () => {
  try {
    // 使用本地模型
    model.value = await tf.loadLayersModel('/djtu-wiki/model/model.json')
    console.log('Model loaded successfully')
  } catch (error) {
    console.error('Error loading model:', error)
  }
}

// 预处理画布数据
const preprocessCanvas = () => {
  const imageData = ctx.value.getImageData(0, 0, canvasRef.value.width, canvasRef.value.height)
  return tf.tidy(() => {
    // 转换为张量
    let tensor = tf.browser.fromPixels(imageData, 1)
    // 调整大小为 28x28
    tensor = tf.image.resizeBilinear(tensor, [28, 28])
    // 归一化到 0-1
    tensor = tensor.toFloat().div(255.0)
    // 反转颜色（因为 MNIST 模型期望黑底白字）
    tensor = tensor.mul(-1).add(1)
    // 添加批次维度
    tensor = tensor.expandDims(0)
    return tensor
  })
}

// 绘图相关函数
const startDrawing = (e) => {
  isDrawing.value = true
  draw(e)
}

const stopDrawing = () => {
  isDrawing.value = false
  ctx.value.beginPath()
}

const draw = (e) => {
  if (!isDrawing.value) return
  
  const rect = canvasRef.value.getBoundingClientRect()
  let x, y
  
  if (e.touches) {
    x = e.touches[0].clientX - rect.left
    y = e.touches[0].clientY - rect.top
  } else {
    x = e.clientX - rect.left
    y = e.clientY - rect.top
  }
  
  // 调整坐标以适应画布实际大小
  x = (x / rect.width) * canvasRef.value.width
  y = (y / rect.height) * canvasRef.value.height
  
  ctx.value.lineWidth = 20
  ctx.value.lineCap = 'round'
  ctx.value.strokeStyle = '#fff'
  
  ctx.value.lineTo(x, y)
  ctx.value.stroke()
  ctx.value.beginPath()
  ctx.value.moveTo(x, y)
}

// 清除画布
const clearCanvas = () => {
  ctx.value.fillStyle = '#000'
  ctx.value.fillRect(0, 0, canvasRef.value.width, canvasRef.value.height)
  predictedNumber.value = null
  activations.value = new Array(networkLayers.value.length).fill(0)
  isStarted.value = false
  isPredicting.value = false
  drawNetwork()
  drawResult()
  
  // 清除结果画布
  const resultCtx = resultCanvasRef.value.getContext('2d')
  resultCtx.clearRect(0, 0, resultCanvasRef.value.width, resultCanvasRef.value.height)
}

// 动画网络
const animateNetwork = () => {
  return new Promise((resolve) => {
    let currentLayer = 0
    activations.value = new Array(networkLayers.value.length).fill(0)
    
    const interval = setInterval(async () => {
      if (currentLayer >= networkLayers.value.length) {
        clearInterval(interval)
        await predictNumber()
        resolve()
        return
      }
      activations.value[currentLayer] = 1
      drawNetwork()
      currentLayer++
    }, 300)
  })
}

// 绘制网络
const drawNetwork = () => {
  const netCtx = networkCanvasRef.value.getContext('2d')
  netCtx.clearRect(0, 0, networkCanvasRef.value.width, networkCanvasRef.value.height)
  
  const width = networkCanvasRef.value.width
  const height = networkCanvasRef.value.height
  const layerGap = width / (networkLayers.value.length + 1)

  networkLayers.value.forEach((layer, i) => {
    const x = layerGap * (i + 1)
    const neuronGap = (height - 60) / (layer.neurons + 1)
    
    // 绘制层名称
    netCtx.font = '12px Arial'
    netCtx.textAlign = 'center'
    netCtx.fillStyle = '#fff'
    netCtx.fillText(layer.name, x, 20)
    
    // 绘制神经元
    for (let j = 0; j < layer.neurons; j++) {
      const y = neuronGap * (j + 1) + 30
      
      // 绘制连接线
      if (i > 0) {
        const prevLayer = networkLayers.value[i - 1]
        const prevX = layerGap * i
        const prevNeuronGap = (height - 60) / (prevLayer.neurons + 1)

        // 减少连接线数量
        if (j % 2 === 0) {
          for (let k = 0; k < prevLayer.neurons; k += 2) {
            const prevY = prevNeuronGap * (k + 1) + 30
            netCtx.beginPath()
            netCtx.moveTo(prevX, prevY)
            netCtx.lineTo(x, y)
            
            if (activations.value[i]) {
              const gradient = netCtx.createLinearGradient(prevX, prevY, x, y)
              gradient.addColorStop(0, 'rgba(72, 175, 232, 0.4)')
              gradient.addColorStop(1, 'rgba(72, 175, 232, 0.1)')
              netCtx.strokeStyle = gradient
            } else {
              netCtx.strokeStyle = 'rgba(51, 51, 51, 0.2)'
            }
            
            netCtx.lineWidth = 0.1
            netCtx.stroke()
          }
        }
      }

      // 绘制神经元
      netCtx.beginPath()
      netCtx.arc(x, y, 2, 0, Math.PI * 2)
      
      if (i === networkLayers.value.length - 1) {
        netCtx.fillStyle = (j === predictedNumber.value && activations.value[i]) 
          ? '#48AFE8' 
          : (activations.value[i] ? '#666' : '#333')
        if (activations.value[i]) {
          netCtx.font = '10px Arial'
          netCtx.fillText(j.toString(), x + 10, y)
        }
      } else {
        netCtx.fillStyle = activations.value[i] ? '#48AFE8' : '#666'
      }
      netCtx.fill()
    }
  })
}

// 预测数字
const predictNumber = async () => {
  if (!model.value) {
    console.error('Model not loaded')
    return
  }

  try {
    const tensor = preprocessCanvas()
    const prediction = await model.value.predict(tensor).data()
    const maxIndex = prediction.indexOf(Math.max(...prediction))
    predictedNumber.value = maxIndex
    
    // 显示预测结果和概率分布
    drawResult()
    drawProbabilities(Array.from(prediction))
    
    // 清理内存
    tensor.dispose()
  } catch (error) {
    console.error('Prediction error:', error)
  }
}

// 绘制概率分布
const drawProbabilities = (probabilities) => {
  const resultCtx = resultCanvasRef.value.getContext('2d')
  const width = resultCanvasRef.value.width
  const height = resultCanvasRef.value.height
  const barWidth = width / 12
  const maxProb = Math.max(...probabilities)

  resultCtx.clearRect(0, 0, width, height)
  
  // 先绘制预测结果
  resultCtx.font = 'bold 48px Arial'
  resultCtx.fillStyle = '#48AFE8'
  resultCtx.textAlign = 'center'
  resultCtx.fillText(
    `预测结果: ${predictedNumber.value}`,
    width / 2,
    60
  )
  
  // 绘制概率条
  probabilities.forEach((prob, i) => {
    const barHeight = (prob / maxProb) * (height * 0.6)  // 减小高度留出空间给预测结果
    const x = barWidth * (i + 1)
    const y = height - barHeight - 40  // 上移概率条

    // 绘制概率条
    resultCtx.fillStyle = i === predictedNumber.value ? '#48AFE8' : '#666'
    resultCtx.fillRect(x, y, barWidth * 0.8, barHeight)

    // 绘制数字标签
    resultCtx.fillStyle = '#fff'
    resultCtx.font = '14px Arial'
    resultCtx.textAlign = 'center'
    resultCtx.fillText(i.toString(), x + barWidth * 0.4, height - 20)
    
    // 绘制概率值
    resultCtx.fillText(
      `${(prob * 100).toFixed(1)}%`,
      x + barWidth * 0.4,
      y - 5
    )
  })
}

// 添加开始预测函数
const startPrediction = async () => {
  if (isPredicting.value) return
  isStarted.value = true
  isPredicting.value = true
  try {
    await animateNetwork()
  } catch (error) {
    console.error('Prediction error:', error)
  } finally {
    isPredicting.value = false
  }
}

// 添加结果显示函数
const drawResult = () => {
  const resultCtx = resultCanvasRef.value.getContext('2d')
  resultCtx.clearRect(0, 0, resultCanvasRef.value.width, resultCanvasRef.value.height)

  if (predictedNumber.value !== null) {
    // 绘制大号预测结果
    resultCtx.font = 'bold 120px Arial'
    resultCtx.fillStyle = '#48AFE8'
    resultCtx.textAlign = 'center'
    resultCtx.textBaseline = 'middle'
    resultCtx.fillText(
      predictedNumber.value.toString(),
      resultCanvasRef.value.width / 2,
      resultCanvasRef.value.height / 2
    )

    // 添加标签
    resultCtx.font = '24px Arial'
    resultCtx.fillStyle = '#48AFE8'
    resultCtx.fillText(
      '预测结果',
      resultCanvasRef.value.width / 2,
      resultCanvasRef.value.height - 40
    )
  }
}

// 在组件挂载时加载模型
onMounted(async () => {
  ctx.value = canvasRef.value.getContext('2d')
  // 设置画布背景
  clearCanvas()
  drawNetwork()
  try {
    await loadModel()
    console.log('Model loaded successfully')
  } catch (error) {
    console.error('Error loading model:', error)
  }
})

// 添加触摸支持
const handleTouchStart = (e) => {
  e.preventDefault()
  isDrawing.value = true
  const touch = e.touches[0]
  draw(touch)
}

const handleTouchMove = (e) => {
  e.preventDefault()
  if (!isDrawing.value) return
  const touch = e.touches[0]
  draw(touch)
}

const handleTouchEnd = (e) => {
  e.preventDefault()
  stopDrawing()
}

// 修改画布尺寸
const canvasSize = {
  width: 280,
  height: 280,
  networkWidth: 500,
  networkHeight: 250,
  resultWidth: 280,
  resultHeight: 280
}
</script>

<template>
  <div class="mnist-container">
    <div class="canvas-section">
      <div class="section-title">输入数字</div>
      <div class="canvas-wrapper">
        <canvas
          ref="canvasRef"
          width="280"
          height="280"
          @mousedown="startDrawing"
          @mousemove="draw"
          @mouseup="stopDrawing"
          @mouseleave="stopDrawing"
          @touchstart="handleTouchStart"
          @touchmove="handleTouchMove"
          @touchend="handleTouchEnd"
        ></canvas>
        <div class="button-group">
          <button @click="clearCanvas" class="control-button">清除</button>
          <button 
            @click="startPrediction" 
            class="control-button primary"
            :disabled="isPredicting || isStarted"
          >
            开始识别
          </button>
        </div>
      </div>
    </div>
    
    <div class="network-section">
      <div class="section-title">神经网络可视化</div>
      <div class="network-wrapper">
        <canvas
          ref="networkCanvasRef"
          :width="canvasSize.networkWidth"
          :height="canvasSize.networkHeight"
        ></canvas>
      </div>
    </div>
    
    <div class="result-section">
      <div class="section-title">预测结果</div>
      <div class="result-wrapper">
        <canvas
          ref="resultCanvasRef"
          width="200"
          height="300"
        ></canvas>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mnist-container {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin: 2rem auto;
  gap: 2rem;
  max-width: 1400px;
  padding: 0 2rem;
}

.canvas-section,
.network-section,
.result-section {
  flex: 0 0 auto;
  width: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.network-section {
  width: 450px;
}

.section-title {
  font-size: 1.2rem;
  font-weight: bold;
  margin-bottom: 1rem;
  color: #48AFE8;
}

.canvas-wrapper,
.network-wrapper,
.result-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.canvas-wrapper canvas,
.network-wrapper canvas,
.result-wrapper canvas {
  border: 2px solid #48AFE8;
  border-radius: 8px;
  background-color: #1a1a1a;
}

.network-wrapper canvas {
  width: 100%;
  max-width: 500px;
  height: 250px;
  background-color: rgba(26, 26, 26, 0.8);
}

.button-group {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1rem;
}

.control-button {
  padding: 8px 16px;
  background-color: #666;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.control-button:hover:not(:disabled) {
  background-color: #777;
  transform: translateY(-2px);
}

.control-button.primary {
  background-color: #48AFE8;
}

.control-button.primary:hover:not(:disabled) {
  background-color: #3b82f6;
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

@media (max-width: 1400px) {
  .mnist-container {
    flex-direction: column;
    align-items: center;
    gap: 3rem;
  }

  .canvas-section,
  .network-section,
  .result-section {
    width: 100%;
    max-width: 450px;
  }

  .network-wrapper canvas {
    width: 100%;
  }
}

.canvas-wrapper canvas {
  width: 280px;
  height: 300px;
}

.result-wrapper canvas {
  width: 280px;
  height: 300px;
}
</style> 