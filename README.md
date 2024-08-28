# autograd

Automatic gradient calculation based on computational graph, implemented backpropagation algorithm and gradient descent algorithm.

## Usage

gradient calculation

```javascript
let x = Var()
let y = Var()
let z = Var()
let r = x.mul(y).add(z).add(x.add(y).mul(z)) // r(x, y, z) = xyz + x + y + z

r.eval([x, y, z], [5, 4, 7])

console.log(r.value) // r(5, 4, 7) = 90
console.log(x.grad)  // dr/dx = 11
console.log(y.grad)  // dr/dy = 12
console.log(z.grad)  // dr/dz = 10
```

gradient descent algorithm to calculate minimal value of function

```javascript
let x = Var(10)
let y = Var(20)
let z = x.mul(x).add(Const(4).mul(y).mul(y)) // z(x, y) = x^2 + 4y^2

let endCondition = (epoch, target, vars) => target.value < 1e-3
train(z, [x, y], AdamOptimizer(), endCondition)
console.log(z.value, x.value, y.value) 
// 0.0009988579447520486 1.1315571976043337e-10 0.015802356982045816
```