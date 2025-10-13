# autograd

Automatic gradient calculation based on computational graph, implemented backpropagation algorithm and gradient descent algorithm.

## Usage

Calculate the gradient of a multivariable function:

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

Use gradient descent to calculate the minimum of a function:

```javascript
let x = Var(10)
let y = Var(20)
let z = x.mul(x).add(Const(4).mul(y).mul(y)) // z(x, y) = x^2 + 4y^2

let endCondition = (epoch, target, vars) => target.value < 1e-3
train(z, [x, y], AdamOptimizer(), endCondition)
console.log(z.value, x.value, y.value) 
```

Fit a quadratic curve using gradient descent:

```javascript
let points = [
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

// ax^2+by^2+cxy+dx+ey+f=0
let a = Var(Math.random())
let b = Var(Math.random())
let c = Var(Math.random())
let d = Var(Math.random())
let e = Var(Math.random())
let f = Var(Math.random())

let loss = Const(0)
for (let [x, y] of points) {
    loss = loss.add(
        a.mul(Const(x * x))
        .add(b.mul(Const(y * y)))
        .add(c.mul(Const(x * y)))
        .add(d.mul(Const(x)))
        .add(e.mul(Const(y)))
        .add(f)
        .pow(Const(2))
    )
}

let endCondition = (epoch, target, vars) => {
    return target.value < 1e-6 || epoch > 50000
}
train(loss, [a, b, c, d, e, f], AdamOptimizer(), endCondition)

console.log(`(${a.value})x^2+(${b.value})y^2+(${c.value})xy+(${d.value})x+(${e.value})y+(${f.value})=0`)
console.log(loss.value)
```
