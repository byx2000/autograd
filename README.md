# autograd

Automatic gradient calculation based on computational graph, implemented backpropagation algorithm and gradient descent algorithm.

## Usage

Calculate the gradient of a multivariable function:

```javascript
import { Var } from "./autograd.js"

const x = Var()
const y = Var()
const z = Var()
const r = x.mul(y).add(z).add(x.add(y).mul(z)) // r(x, y, z) = xyz + x + y + z
const graph = r.toComputeGraph()

graph.eval([x, y, z], [5, 4, 7])

console.log(r.value) // r(5, 4, 7) = 90
console.log(x.grad)  // dr/dx = 11
console.log(y.grad)  // dr/dy = 12
console.log(z.grad)  // dr/dz = 10
```

Use gradient descent to calculate the minimum of a function:

```javascript
import { Const, Var } from "./autograd.js"
import { AdamOptimizer, train } from "./train.js"

const x = Var(10)
const y = Var(20)
const z = x.add(Const(1)).square().add(Const(4).mul(y.sub(Const(2)).square())) // z(x, y) = (x+1)^2 + 4*(y-2)^2

const endCondition = (epoch, target, vars) => target.value < 1e-8
train(z.toComputeGraph(), [x, y], AdamOptimizer(), endCondition)
console.log(z.value, x.value, y.value)
```

Fit a quadratic curve using gradient descent:

```javascript
import { Const, Var } from "./autograd.js"
import { AdamOptimizer, train } from "./train.js"

const points = [
  [1.96, 4.26],
  [0.75, 3.86],
  [-0.35, 3.1],
  [-1.11, 2.48],
  [-1.73, 1.85],
  [-2.39, 0.93],
  [-2.84, 0.11],
  [-3.17, -1.38],
  [-3.17, -2.48],
  [-2.81, -3.36],
  [-2.04, -3.64],
  [-0.53, -3.82],
  [0.33, -3.61],
  [0.99, -3.34],
  [1.95, -2.86],
  [2.89, -2.15],
  [3.45, -1.36],
  [4.08, -0.46],
  [4.27, 0.3],
  [4.52, 1.27],
  [4.32, 2.42],
  [3.93, 3.41],
  [3.17, 4.12],
]

// x^2+ay^2+bxy+cx+dy+e=0
const a = Var(Math.random())
const b = Var(Math.random())
const c = Var(Math.random())
const d = Var(Math.random())
const e = Var(Math.random())

let loss = Const(0)
for (const [x, y] of points) {
  loss = loss.add(
    Const(x * x)
      .add(a.mul(Const(y * y)))
      .add(b.mul(Const(x * y)))
      .add(c.mul(Const(x)))
      .add(d.mul(Const(y)))
      .add(e)
      .square()
  )
}

const endCondition = (epoch, target, vars) => {
  return target.value < 1e-3 || epoch > 10000
}
train(loss.toComputeGraph(), [a, b, c, d, e], AdamOptimizer(), endCondition)

console.log(`x^2+(${a.value})y^2+(${b.value})xy+(${c.value})x+(${d.value})y+(${e.value})=0`)
console.log(loss.value)
```
