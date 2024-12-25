---
outline: deep
---

# 代加工没写完()

<script setup>
import { defineClientComponent } from 'vitepress'
const MnistVisualizer = defineClientComponent(() => import('./MnistVisualizer.vue'))
</script>

## MNIST 手写数字识别可视化

<ClientOnly>
  <MnistVisualizer />
</ClientOnly>

## 工作原理

这个演示使用了预训练的卷积神经网络模型来识别手写数字：

1. **输入预处理**：
   - 将手写输入缩放到 28x28 像素
   - 转换为灰度图像
   - 标准化像素值到 0-1 范围

2. **网络结构**：
   - 输入层 (28×28=784 个神经元)
   - 卷积层 1 (32 个特征图)
   - 池化层 1 (16×16 降采样)
   - 卷积层 2 (64 个特征图)
   - 池化层 2 (8×8 降采样)
   - 全连接层 (128 个神经元)
   - 输出层 (10 个神经元，对应 0-9)

3. **实时预测**：
   - 每次绘制完成后自动预测
   - 显示每个数字的概率分布
   - 动态可视化网络激活状态

4. **技术栈**：
   - TensorFlow.js 用于模型加载和预测
   - Canvas API 用于绘图和可视化
   - Vue 3 用于响应式 UI
