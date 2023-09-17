'use strict'

const assert = require('assert')
const { Var, Const } = require("../src/autograd")
const { train, SGDOptimizer, MomentumOptimizer, AdaGradOptimizer, RMSpropOptimizer, AdamOptimizer } = require("../src/train")

let endCondition = (epoch, target, vars) => target.value < 1e-3

describe('test of train and optimizer', () => {
  it('SGDOptimizer', () => {
    let x = Var(10)
    let y = Var(20)
    let z = x.mul(x).add(Const(4).mul(y).mul(y))
    train(z, [x, y], SGDOptimizer(), endCondition)
    assert.equal(z.value < 1e-3, true)
  })
  it('MomentumOptimizer', () => {
    let x = Var(10)
    let y = Var(20)
    let z = x.mul(x).add(Const(4).mul(y).mul(y))
    train(z, [x, y], MomentumOptimizer(), endCondition)
    assert.equal(z.value < 1e-3, true)
  })
  it('AdaGradOptimizer', () => {
    let x = Var(10)
    let y = Var(20)
    let z = x.mul(x).add(Const(4).mul(y).mul(y))
    train(z, [x, y], AdaGradOptimizer(1), endCondition)
    assert.equal(z.value < 1e-3, true)
  })
  it('RMSpropOptimizer', () => {
    let x = Var(10)
    let y = Var(20)
    let z = x.mul(x).add(Const(4).mul(y).mul(y))
    train(z, [x, y], RMSpropOptimizer(), endCondition)
    assert.equal(z.value < 1e-3, true)
  })
  it('AdamOptimizer', () => {
    let x = Var(10)
    let y = Var(20)
    let z = x.mul(x).add(Const(4).mul(y).mul(y))
    train(z, [x, y], AdamOptimizer(), endCondition)
    assert.equal(z.value < 1e-3, true)
  })
})


