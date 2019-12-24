@file:Suppress("MemberVisibilityCanBePrivate")

import arrow.core.None
import arrow.core.Option
import arrow.core.getOrElse
import kotlinx.collections.immutable.PersistentList
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.dot
import org.jetbrains.numkt.core.max
import org.jetbrains.numkt.math.divAssign
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.math.plusAssign
import org.jetbrains.numkt.math.plus
import org.jetbrains.numkt.math.sum
import org.jetbrains.numkt.math.times
import org.jetbrains.numkt.zeros
import java.lang.IllegalStateException
import kotlin.math.max

typealias Real = Double
typealias Matrix = KtNDArray<Real>
typealias LossFunc = (Int, Matrix) -> Real /* (ExpectedLabel, Actual) -> Loss */
typealias RegularisationFunc = (Matrix) -> Real
/* nth class to Image */
typealias Labelled = Pair<Int, Matrix>

data class Loss(
  val lossFunc: LossFunc,
  val dLossFunc: (Int, Matrix, Real) -> Matrix
)

data class LearnableParams(val w: Matrix, val b: Matrix)
typealias LayersWithParams = Map<LinearClassifier, LearnableParams>

sealed class Layer() {
  // TODO verify this is a 1 column matrix
  var values = zeros<Real>(1, HyperParams.neuronsPerLayer)
  var dValues = zeros<Real>(*values.shape)

  /**
   * Propagates [Layer.values] from [Layer.prev] over to this layer
   */
  abstract fun feedForward()

  /**
   * Uses the cached values in [Layer.values] and [Layer.next]'s derivatives in
   * order to populate this layer's [Layer.dValues].
   */
  abstract fun feedBackward()

  abstract val prev: Option<Layer>
  abstract val next: Option<Layer>
}

class LinearClassifier(var w: Matrix, var b: Matrix) : Layer() {

  override var prev: Option<Layer> = None
  override var next: Option<Layer> = None

  /**
   * Weights derivatives and biases vectors
   */
  var dW = zeros<Real>(*values.shape)
  var dB = zeros<Real>(*values.shape)

  /**
   * Momentum stored vector
   */
  var wVx = zeros<Real>(*values.shape)
  var bVx = zeros<Real>(*values.shape)

  override fun feedForward() {
    assert(prev.isDefined()) { "Called feedForward on a layer without previous layer" }
    val previous = prev.getOrElse { throw IllegalStateException() }
    values = previous.values.dot(w) // TODO + bias
  }

  override fun feedBackward() {
    assert(next.isDefined()) { "Called feedBackward on a layer withput a next layer" }
    val next = this.next.getOrElse { throw IllegalStateException() }
    dW = values.t.dot(next.dValues) + b
    dB = sum(next.dValues, axis = 0)
    dValues = next.dValues.dot(w.t)
    assert(dValues.shape.contentEquals(values.shape))
  }
}

class ReLuActivation() : Layer() {

  override var prev: Option<Layer> = None
  override var next: Option<Layer> = None

  val reLuActivation: (Matrix) -> Matrix = { m ->
    // TODO reall?
    m.map { max(0.0, it) }
  }

  val dReLuActivation: (Matrix, Matrix) -> Matrix = { values, dValues ->
    val ret = zeros<Real>(*dValues.shape)
    for ((i, dV) in dValues.withIndex()) {
      ret[i] = if (values[i] > 0) dV else 0.0
    }
    ret
  }

  override fun feedForward() {
    assert(prev.isDefined()) { "Called feedForward on a layer without previous layer" }
    val previous = prev.getOrElse { throw IllegalStateException() }
    values = reLuActivation(previous.values)
  }

  override fun feedBackward() {
    assert(next.isDefined()) { "Called feedBackward on a layer withput a next layer" }
    val next = this.next.getOrElse { throw IllegalStateException() }
    dValues = dReLuActivation(values, next.values)
  }
}

class NeuralNet(
  val lossFuncs: Loss,
  val regularisation: Option<RegularisationFunc> = None,
  val input: Layer,
  val middleLayers: PersistentList<Layer>,
  val output: Layer,
  val trainingData: Collection<Labelled>
) {

  /**
   * Performs a forward pass with single input [x] using weights [w] and TODO biases.
   * Returns the resulting output matrix (1 column, classes lines)
   */
  fun forwardPass(x: Matrix): Matrix =
    (middleLayers + output)
      .also { input.values = x }
      .onEach { it.feedForward() }
      .last()
      .values

  /**
   * Performs a single back propagation with weights [w] and input left from the latest
   * [NeuralNet.forwardPass]. Returns dW, the matrix of the gradients of each weight.
   *
   * TODO biases too rip
   */
  fun backwardProp(dscores: Matrix) =
    (input + middleLayers)
      // Put dscores on the last layer
      .also { output.dValues = dscores }
      // Propagate back
      .onEach { it.feedBackward() }
      .filterIsInstance<LinearClassifier>()
      // Return the weights and biases
      .map { it to (LearnableParams(it.dW, it.dB)) }
      .toMap()

  /**
   * Evaluates the gradient for every x in [xs].
   * For this, for each x, a forward and backward pass is performed.
   *
   * @return A map of each layer's derivative of its parameters, with the total loss, with the
   * total batch accuracy at predicting.
   *
   * TODO refactor??
   */
  fun evalAvgGradient(xs: Collection<Labelled>): Triple<LayersWithParams, Real, Real> {
    val gradientSum = HashMap<LinearClassifier, LearnableParams>()
    val batchSize = xs.size
    var lossSum = 0.0
    var rightGuesses = 0
    var wrongGuesses = 0
    for ((label, x) in xs) {
      val scores = forwardPass(x)

      val predicted = scores.toList().withIndex().maxBy { it.value }!!.index
      if (label == predicted) rightGuesses++
      else wrongGuesses++

      val loss = lossFuncs.lossFunc(label, scores)
      lossSum += loss

      val curriedLoss = { ss: Matrix -> lossFuncs.lossFunc(label, ss) }

      // TODO define dLossDunc
      val dscores = curriedLoss.numericalGradient(scores)
      //  lossFuncs.dLossFunc(label, scores, loss)
      val paramGradients = backwardProp(dscores)

      if (gradientSum.isEmpty())
        gradientSum.putAll(paramGradients)
      else
        paramGradients.forEach { i, (dW, dB) ->
          val (dWSum, dBSum) = gradientSum[i]!!
          dWSum += dW
          dBSum += dB
        }
    }
    val totalLoss = lossSum / batchSize
    val accuracy = rightGuesses * 100.0 / batchSize
    gradientSum.values.onEach { (dW, dB) ->
      dW /= batchSize
      dB /= batchSize
    }
    return Triple(gradientSum, totalLoss, accuracy)
  }
}

/**
 * Runs every x in [batch] through [net] and returns
 * the success rate and the average gradients of the weights of the batch
 */
fun trainBatch(
  validation: Collection<Labelled>,
  batch: Collection<Labelled>,
  net: NeuralNet
) {
  val rho = HyperParams.frictionCoeff
  val stepSize = HyperParams.learningRate
  val linearClassifiers = (net.input + net.middleLayers).filterIsInstance<LinearClassifier>()
  var iterationCount = 0
  while (true /* TODO change */) {
    val (grads, loss, accuracy) = net.evalAvgGradient(batch)

    if (iterationCount % 100 == 0) {
      println(
        "Iteration $iterationCount:\n" +
          "  Batch avg loss: $loss\n" +
          "  Batch avg accuracy: $accuracy"
      )
    }
    iterationCount++

    linearClassifiers
      .forEach { layer ->
        val (wdx, bdx) = grads[layer]!!

        // Update weights
        layer.wVx = layer.wVx * rho - wdx
        layer.w = layer.w + layer.wVx

        // Update biases
        layer.bVx = layer.bVx * rho - bdx
        layer.b = layer.b + layer.bVx
      }
  }
}
