'use strict'

class ComputeNode {
  constructor(evalFunc, diffFunc, ...subNodes) {
    this.value = 0
    this.grad = 0
    this.children = []
    this.parents = []
    this.eval = evalFunc
    this.diff = diffFunc

    for (let node of subNodes) {
      this.children.push(node)
      node.parents.push(this)
    }
  }

  forward() {
    _forward(this, new Set())
  }

  backward() {
    let inDegree = new Map()
    _calculateInDegree(this, inDegree)

    this.grad = 1

    // 按拓扑排序处理节点
    let ready = []
    ready.push(this)
    while (ready.length > 0) {
        // 获取当前节点
        let cur = ready.shift()

        // 更新节点出度，添加出度为0的节点到就绪队列
        for (let c of cur.children) {
            let val = inDegree.get(c);
            inDegree.set(c, val - 1);
            if (val === 1 && c.children.length > 0) {
                ready.push(c);
            }
        }

        // 计算本地梯度
        let input = [];
        for (let c of cur.children) {
            input.push(c.value);
        }
        let localGrad = cur.diff(input);

        // 更新子节点梯度值
        for (let i = 0; i < localGrad.length; ++i) {
            cur.children[i].grad += localGrad[i] * cur.grad;
        }
    }
  }

  add(rhs) {
    return Add(this, rhs)
  }

  sub(rhs) {
    return Sub(this, rhs)
  }

  mul(rhs) {
    return Mul(this, rhs)
  }

  div(rhs) {
    return Div(this, rhs)
  }

  pow(rhs) {
    return Pow(this, rhs)
  }
}

function _forward(node, visited) {
  if (visited.has(node)) {
    return
  }

  visited.add(node)
  let p = []
  for (let c of node.children) {
    _forward(c, visited)
    p.push(c.value)
  }

  node.value = node.eval(p)
}

function _calculateInDegree(node, inDegree) {
  if (inDegree.has(node)) {
      return
  }
  node.grad = 0
  inDegree.set(node, node.parents.length)
  for (let c of node.children) {
    _calculateInDegree(c, inDegree)
  }
}

function Const(val) {
  return new ComputeNode(p => val, p => [])
}

function Var(val = 0) {
  let node = new ComputeNode(p => node.value, p => [])
  node.value = val
  return node
}

function BinaryOp(evalFunc, diffFunc) {
  return (lhs, rhs) => new ComputeNode(p => evalFunc(p[0], p[1]), p => diffFunc(p[0], p[1]), lhs, rhs)
}

let Add = BinaryOp((a, b) => a + b, (a, b) => [1, 1])
let Sub = BinaryOp((a, b) => a - b, (a, b) => [1, -1])
let Mul = BinaryOp((a, b) => a * b, (a, b) => [b, a])
let Div = BinaryOp((a, b) => a / b, (a, b) => [1 / b, -a/(b * b)])
let Pow = BinaryOp((a, b) => Math.pow(a, b), (a, b) => [b * Math.pow(a, b - 1), Math.log(a) * Math.pow(a, b)])

function UnaryOp(evalFunc, diffFunc) {
  return n => new ComputeNode(p => evalFunc(p[0]), p => [diffFunc(p[0])], n)
}

let Neg = UnaryOp(x => -x, x => -1)
let Sin = UnaryOp(x => Math.sin(x), x => Math.cos(x))
let Cos = UnaryOp(x => Math.cos(x), x => -Math.sin(x))
let Tan = UnaryOp(x => Math.tan(x), x => 1 / (Math.cos(x) * Math.cos(x)))
let Exp = UnaryOp(x => Math.exp(x), x => Math.exp(x))
let Ln = UnaryOp(x => Math.log(x), x => 1 / x)

module.exports = {
  ComputeNode, UnaryOp, BinaryOp,
  Const, Var,
  Add, Sub, Mul, Div, Pow,
  Neg, Sin, Cos, Tan, Exp, Ln
}
