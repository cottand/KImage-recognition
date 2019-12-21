import arrow.core.Option
import arrow.core.getOrElse
import arrow.core.toOption
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.copy
import org.jetbrains.numkt.linalg.dot
import org.jetbrains.numkt.zeros
import java.lang.IllegalStateException

typealias Real = Double
typealias Matrix = KtNDArray<Real>
typealias MatrixInts = KtNDArray<Int>
typealias ErrorFunc = (Real, Real) -> Real
typealias RegularisationFunc = (Matrix) -> Real

object HyperParams {
  const val regularisationCoeff: Real = 0.1
  const val gradientH = 0.00001
  const val neuronsPerLayer = 5
}

data class Activation(
  val activationFunc: (Matrix) -> Matrix,
  val dActivationFUnc: (Double) -> Double
)

sealed class Layer(val weights: Matrix) {
  // TODO verify this is a 1 column matrix
  var values = zeros<Real>(1, HyperParams.neuronsPerLayer)
  var derivatives = values.copy()
  abstract fun feedForward()
  abstract fun feedBackward()
  abstract val prev: Option<Layer>
  abstract val next: Option<Layer>
}

class LinearClassifier(
  weights: Matrix,
  val activation: Activation,
  previous: Layer? = null,
  next: Layer? = null
) : Layer(weights) {
  override val prev: Option<Layer> = previous.toOption()
  override val next: Option<Layer> = next.toOption()

  override fun feedForward() {
    assert(prev.isDefined()) {"Called feedForward on a layer without previous layer"}
    val previous = prev.getOrElse { throw IllegalStateException() }
    // TODO get nth column of weights
    values = activation.activationFunc(dot(previous.values, weights[0..1]))
  }

  override fun feedBackward() {
    TODO("not implemented")
  }
}

class NeuralNet(
  val error: ErrorFunc,
  val regularisation: RegularisationFunc,
  val w: Matrix,
  val input: Layer
) {

}
