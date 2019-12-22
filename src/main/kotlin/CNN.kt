@file:Suppress("MemberVisibilityCanBePrivate")

import arrow.core.None
import arrow.core.Option
import arrow.core.getOrElse
import kotlinx.collections.immutable.PersistentList
import org.jetbrains.numkt.array
import org.jetbrains.numkt.concatenate
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.math.plusAssign
import org.jetbrains.numkt.math.times
import org.jetbrains.numkt.zeros
import java.lang.IllegalStateException

typealias Real = Double
typealias Matrix = KtNDArray<Real>
typealias LossFunc = (Int, Matrix) -> Real /* (ExpectedLabel, Actual) -> Loss */
typealias RegularisationFunc = (Matrix) -> Real
/* nth class to Image */
typealias Labelled = Pair<Int, Matrix>

fun Labelled.asMatrix(): Pair<Matrix, Matrix> {
  val list = (0..9).map { HyperParams.minScore }.toMutableList()
  list[this.first] = HyperParams.maxScore
  // TODO verify this is a 1 column 10 line array
  val left = array<Real>(listOf(list)).t
  // TODO verify this is a 2 column 10 line array with expected scores on the left
  return left to this.second
}

fun train() {
}

/**
 * Stores hyper parameters and practical constants.
 */
object HyperParams {
  const val regularisationCoeff: Real = 0.1
  const val gradientH = 0.00001
  const val neuronsPerLayer = 5
  const val learningRate = 1e-7 /* e-4 .. e-11 */
  const val frictionCoeff: Real = 0.95 /* 0.9 .. 0.99 */
  const val batchSize = 1000
  const val classes = 10
  const val minScore = 0.0
  const val maxScore = 10.0
}

data class Activation(
  val activationFunc: (Matrix) -> Matrix,
  val dActivationFUnc: (Matrix) -> Matrix
)

// TODO maybe reals instead of matrices
data class Derivatives(val dx: Matrix, val dw: Matrix)

data class Loss(
  val lossFunc: LossFunc,
  val dLossFunc: LossFunc
)

sealed class Layer() {
  // TODO verify this is a 1 column matrix
  var values = zeros<Real>(1, HyperParams.neuronsPerLayer)
  var dValues = zeros<Real>(*values.shape)
  var dW = zeros<Real>(*values.shape)

  /**
   * Propagates [Layer.values] from [Layer.prev] over to this layer
   */
  abstract fun feedForward(w: Matrix)

  /**
   * Uses the cached values in [Layer.values] and [Layer.next]'s derivatives in
   * order to populate this layer's [Layer.dValues].
   *
   * This means this [Layer.dValues] are the partial derivative of the input
   * with respect to [w]
   */
  abstract fun feedBackward(w: Matrix)

  abstract val prev: Option<Layer>
  abstract val next: Option<Layer>
}

class LinearClassifier(val activation: Activation) : Layer() {

  override var prev: Option<Layer> = None
  override var next: Option<Layer> = None

  override fun feedForward(w: Matrix) {
    assert(prev.isDefined()) { "Called feedForward on a layer without previous layer" }
    val previous = prev.getOrElse { throw IllegalStateException() }
    // TODO get nth column of weights
    values = activation.activationFunc(previous.values dot w[0..1])
  }

  override fun feedBackward(w: Matrix) {
    assert(next.isDefined()) { "Called feedBackward on a layer withput a next layer" }
    val next = this.next.getOrElse { throw IllegalStateException() }
    dW = values.t dot next.dValues
    // TODO grab this layer's nth w column
    dValues = next.dValues dot w[0..1].t
  }
}

class NeuralNet(
  val lossFuncs: Loss,
  val regularisation: Option<RegularisationFunc> = None,
  val initW: Matrix,
  val input: Layer,
  val middleLayers: PersistentList<Layer>,
  val output: Layer, // TODO training datatype
  val trainingData: Collection<Labelled>
) {

  /**
   * Performs a forward pass with single input [x] using weights [w] and TODO biases.
   * Returns the resulting output matrix (1 column, classes lines)
   */
  fun forwardPass(x: Matrix, w: Matrix): Matrix {
    input.values = x
    (middleLayers + output).forEach { it.feedForward(w) }
    return output.values
  }

  /**
   * Performs a single back propagation with weights [w] and input left from the latest
   * [NeuralNet.forwardPass]. Returns dW, the matrix of the gradients of each weight.
   *
   * TODO biases too rip
   */
  fun backwardProp(w: Matrix): Matrix {
    // TODO put dscores (derivatives of scores with respect to loss) in output layer
    val firstLayers = input + middleLayers
    firstLayers.forEach { it.feedBackward(w) }
    val ws = firstLayers.map { it.dW }.toTypedArray()
    val dW = concatenate(*ws, axis = 1)
    // TODO refactor
    assert(dW.shape.contentEquals(w.shape))
    return dW
  }

  /**
   * Evaluates the gradient for every x in [xs] using the weights [w].
   * For this, for each x, a forward and backward pass is performed.
   */
  fun evalGradient(xs: Collection<Labelled>, w: Matrix): Matrix {
    for ((label, x) in xs) {
      val forwardResult = forwardPass(x, w)
    }
    TODO()
  }
}

fun trainMNIST(initW: Matrix, training: List<Labelled>, validation: Set<Labelled>) {
  val batchSize = HyperParams.batchSize
  val batchNo = training.size / batchSize
  val batches = training.splitIntoBatches(batchSize)
  assert(batches.size in (batchNo - 1)..(batchNo + 1))
}

/**
 * Runs every x in [batch] through [net] with the given [weights] and returns
 * the success rate and the average gradients of the weights of the batch
 *
 */
fun trainBatch(
  weights: Matrix,
  validation: Collection<Labelled>,
  batch: Collection<Labelled>,
  net: NeuralNet
) {
  var vx = zeros<Real>(*weights.shape)
  val rho = HyperParams.frictionCoeff
  val stepSize = HyperParams.learningRate
  while (true /* TODO change */) {
    /* STG velocity momentum */
    val dx = net.evalGradient(batch, weights)
    vx = vx * rho - dx * stepSize
    weights += vx
  }
}
