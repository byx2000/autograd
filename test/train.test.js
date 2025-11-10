'use strict'

import { equal } from 'assert'
import { Var, Const } from '../src/autograd.js'
import { train, SGDOptimizer, MomentumOptimizer, AdaGradOptimizer, RMSpropOptimizer, AdamOptimizer } from '../src/train.js'

let endCondition = (epoch, target, vars) => target.value < 1e-3

describe('test of train and optimizer', () => {
  it('SGDOptimizer', () => {
    const x = Var(10)
    const y = Var(20)
    const z = x.mul(x).add(Const(4).mul(y).mul(y))
    const graph = z.toComputeGraph()
    train(graph, [x, y], SGDOptimizer(), endCondition)
    equal(z.value < 1e-3, true)
  })
  it('MomentumOptimizer', () => {
    const x = Var(10)
    const y = Var(20)
    const z = x.mul(x).add(Const(4).mul(y).mul(y))
    const graph = z.toComputeGraph()
    train(graph, [x, y], MomentumOptimizer(), endCondition)
    equal(z.value < 1e-3, true)
  })
  it('AdaGradOptimizer', () => {
    const x = Var(10)
    const y = Var(20)
    const z = x.mul(x).add(Const(4).mul(y).mul(y))
    const graph = z.toComputeGraph()
    train(graph, [x, y], AdaGradOptimizer(1), endCondition)
    equal(z.value < 1e-3, true)
  })
  it('RMSpropOptimizer', () => {
    const x = Var(10)
    const y = Var(20)
    const z = x.mul(x).add(Const(4).mul(y).mul(y))
    const graph = z.toComputeGraph()
    train(graph, [x, y], RMSpropOptimizer(), endCondition)
    equal(z.value < 1e-3, true)
  })
  it('AdamOptimizer', () => {
    const x = Var(10)
    const y = Var(20)
    const z = x.mul(x).add(Const(4).mul(y).mul(y))
    const graph = z.toComputeGraph()
    train(graph, [x, y], AdamOptimizer(), endCondition)
    equal(z.value < 1e-3, true)
  })
})
