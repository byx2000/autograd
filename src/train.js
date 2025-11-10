'use strict'

/**
 * 训练
 * @param {*} graph 计算图
 * @param {*} vars 变量
 * @param {*} optimizer 优化器
 * @param {*} endCondition 结束条件
 */
function train(graph, vars, optimizer, endCondition) {
  let epoch = 1
  const target = graph.target
  while (true) {
    graph.forward()
    graph.backward()
    if (endCondition(epoch, target, vars)) {
      break
    }
    optimizer(epoch, target, vars)
    epoch++
  }
}

/**
 * 标准梯度下降
 * @param {*} lr 学习率 
 */
function SGDOptimizer(lr = 0.01) {
  let lastValue = undefined
  return (epoch, target, vars) => {
    for (let v of vars) {
      v.value -= lr * v.grad
    }

    if (lastValue !== undefined && target.value >= lastValue) {
      lr /= 2
    }

    lastValue = target.value
  }
}

/**
 * 动量梯度下降
 * @param {*} lr 学习率 
 * @param {*} momentum 动量
 */
function MomentumOptimizer(lr = 0.01, momentum = 0.9) {
  let v = undefined
  return (epoch, target, vars) => {
    if (v === undefined) {
      v = new Array(vars.length).fill(0)
    }
    
    for (let i = 0; i < vars.length; i++) {
      v[i] = momentum * v[i] + (1 - momentum) * vars[i].grad
      vars[i].value -= lr * v[i]
    }
  }
}

/**
 * AdaGrad梯度下降
 * @param {*} lr 学习率
 */
function AdaGradOptimizer(lr = 0.01) {
  let s = undefined
  return (epoch, target, vars) => {
    if (s === undefined) {
      s = new Array(vars.length).fill(0)
    }

    for (let i = 0; i < vars.length; i++) {
      s[i] += vars[i].grad * vars[i].grad
      vars[i].value -= lr * vars[i].grad / (Math.sqrt(s[i]) + 1e-8)
    }
  }
}

/**
 * RMSprop梯度下降
 * @param {*} lr 学习率
 * @param {*} beta 梯度平方的指数平均系数
 */
function RMSpropOptimizer(lr = 0.01, beta = 0.9) {
  let s = undefined
  return (epoch, target, vars) => {
    if (s === undefined) {
      s = new Array(vars.length).fill(0)
    }

    for (let i = 0; i < vars.length; i++) {
      s[i] = beta * s[i] + (1 - beta) * vars[i].grad * vars[i].grad
      vars[i].value -= lr * vars[i].grad / (Math.sqrt(s[i]) + 1e-8)
    }
  }
}

/**
 * Adam梯度下降
 * @param {*} lr 学习率 
 * @param {*} beta1 梯度的指数平均系数
 * @param {*} beta2 梯度平方的指数平均系数
 * @returns 
 */
function AdamOptimizer(lr = 0.01, beta1 = 0.9, beta2 = 0.999) {
  let beta1n = 1
  let beta2n = 1
  let v = undefined
  let s = undefined
  return (epoch, target, vars) => {
    if (v === undefined) {
      v = new Array(vars.length).fill(0)
    }

    if (s === undefined) {
      s = new Array(vars.length).fill(0)
    }

    for (let i = 0; i < vars.length; i++) {
      v[i] = beta1 * v[i] + (1 - beta1) * vars[i].grad
      s[i] = beta2 * s[i] + (1 - beta2) * vars[i].grad * vars[i].grad
      beta1n *= beta1
      beta2n *= beta2
      let vp = v[i] / (1 - beta1n)
      let sp = s[i] / (1 - beta2n)
      vars[i].value -= lr * vp / (Math.sqrt(sp) + 1e-8)
    }
  }
}

export {
  train, 
  SGDOptimizer, MomentumOptimizer, AdaGradOptimizer, RMSpropOptimizer, AdamOptimizer
}
