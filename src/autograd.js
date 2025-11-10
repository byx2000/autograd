'use strict'

// 计算节点
class ComputeNode {
  constructor(evalFunc, diffFunc, ...subNodes) {
    this.value = 0
    this.grad = 0
    this.children = []
    this.parents = []
    this.evalFunc = evalFunc
    this.diffFunc = diffFunc

    for (let node of subNodes) {
      this.children.push(node)
      node.parents.push(this)
    }
  }

  toComputeGraph() {
    return new ComputeGraph(this)
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

  square() {
    return Square(this)
  }
}

// 计算图
class ComputeGraph {
  constructor(root) {
    this.target = root
    this.sortedNodes = []

    // 拓扑排序
    const ready = [root]
    const inDegree = new Map()
    while (ready.length > 0) {
      const cur = ready.shift()
      this.sortedNodes.push(cur)
      for (const c of cur.children) {
        let val = inDegree.get(c)
        if (val === undefined) {
          val = c.parents.length
        }
        inDegree.set(c, val - 1)
        if (val === 1) {
          ready.push(c)
        }
      }
    }
  }

  forward() {
    for (let i = this.sortedNodes.length - 1; i >= 0; --i) {
      const node = this.sortedNodes[i]
      node.value = node.evalFunc(node.children.map(c => c.value))
    }
  }

  backward() {
    for (const node of this.sortedNodes) {
      node.grad = 0
    }
    this.sortedNodes[0].grad = 1

    for (const node of this.sortedNodes) {
      const localGrad = node.diffFunc(node.children.map(c => c.value))
      for (let i = 0; i < node.children.length; i++) {
        node.children[i].grad += localGrad[i] * node.grad
      }
    }
  }

  eval(vars, vals) {
    for (let i = 0; i < vars.length; i++) {
      vars[i].value = vals[i]
    }
    this.forward()
    this.backward()
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

const Add = BinaryOp((a, b) => a + b, (a, b) => [1, 1])
const Sub = BinaryOp((a, b) => a - b, (a, b) => [1, -1])
const Mul = BinaryOp((a, b) => a * b, (a, b) => [b, a])
const Div = BinaryOp((a, b) => a / b, (a, b) => [1 / b, -a/(b * b)])
const Pow = BinaryOp(Math.pow, (a, b) => [b * Math.pow(a, b - 1), Math.log(a) * Math.pow(a, b)])

function UnaryOp(evalFunc, diffFunc) {
  return n => new ComputeNode(p => evalFunc(p[0]), p => [diffFunc(p[0])], n)
}

const Neg = UnaryOp(x => -x, x => -1)
const Sin = UnaryOp(Math.sin, Math.cos)
const Cos = UnaryOp(Math.cos, x => -Math.sin(x))
const Tan = UnaryOp(Math.tan, x => 1 / (Math.cos(x) * Math.cos(x)))
const Exp = UnaryOp(Math.exp, Math.exp)
const Ln = UnaryOp(Math.log, x => 1 / x)
const Square = UnaryOp(x => x * x, x => 2 * x)

export {
  ComputeNode, ComputeGraph,
  UnaryOp, BinaryOp,
  Const, Var,
  Add, Sub, Mul, Div, Pow,
  Neg, Sin, Cos, Tan, Exp, Ln, Square
}
