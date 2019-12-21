import org.jetbrains.numkt.zeros

/**
 * Computes the numerical gradient of receiver at [x]
 */
fun ((Matrix) -> Real).numericalGradient(x: Matrix): Matrix {
  val fx = this(x)
  val grad = zeros<Real>(*x.shape)
  val h = HyperParams.gradientH

  var i = 0
  for (v in x) {
    assert(x[i] == v) // TODO remove

    val oldValue = x[i]
    x[i] = oldValue + h
    val fxh = this(x)

    x[i] = oldValue

    grad[i] = (fxh - fx) / h

    i++
  }
  return grad
}