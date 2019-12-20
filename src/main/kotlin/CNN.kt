import org.jetbrains.numkt.core.KtNDArray

typealias Real = Double
typealias Matrix = KtNDArray<Real>
typealias MatrixInts = KtNDArray<Int>
typealias ErrorFunc = (Real, Real) -> Real
typealias RegularisationFunc = (Matrix) -> Real

object HyperParams {
  const val regularisationCoeff : Real = 0.1
  const val gradientH = 0.00001
}

data class Activation(
  val activationFunc: (Double) -> Double,
  val dActivationFUnc: (Double) -> Double
)

class NeuralNet(
  val error: ErrorFunc,
  val regularisation: RegularisationFunc,
  val w: Matrix
) {

}