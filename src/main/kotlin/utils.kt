import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.math.div
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.zeros

/**
 * Computes the numerical gradient of receiver at [x]
 */
fun ((Matrix) -> Matrix).numericalGradient(x: Matrix): KtNDArray<Matrix> {
  val fx = this(x)
  val grad = zeros<Matrix>(*x.shape)
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