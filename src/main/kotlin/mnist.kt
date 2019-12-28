import arrow.core.None
import kotlinx.collections.immutable.persistentListOf
import org.jetbrains.numkt.math.maximum
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.math.plus
import org.jetbrains.numkt.math.sum
import org.jetbrains.numkt.random.Random
import org.jetbrains.numkt.zeros

/**
 * Stores hyper parameters and practical constants.
 */
object HyperParams {
  const val regularisationCoeff: Real = 0.1
  const val gradientH = 0.00001
  const val neuronsPerLayer = 5
  const val learningRate = 1e-5 /* e-4 .. e-11 */
  const val frictionCoeff: Real = 0.95 /* 0.9 .. 0.99 */
  const val batchSize = 1000
  const val classes = 10
  const val minScore = 0.0
  const val maxScore = 10.0
  const val reportSize: Int = 100
}

val randomMatrix: () -> Matrix = { Random.rand(HyperParams.neuronsPerLayer, 1) }

fun main() {
  val input = LinearClassifier(randomMatrix(), randomMatrix())
  val output = ReLuActivation()
  val middleLayers =
    persistentListOf(ReLuActivation(), LinearClassifier(randomMatrix(), randomMatrix()))
  val loss = Loss(svmLoss, dSvmLoss)
  val data = parseCSVDataMNIST()
  val net = NeuralNet(loss, None, input, middleLayers, output)
  trainBatch(data, net)
}

val svmLoss: LossFunc = { label, actual ->
  val margins = maximum(zeros(*actual.shape), actual - actual[label] + 1)
  margins[label] = 0.0
  sum(margins)
}

// TODO return type??
val dSvmLoss: (Int, Matrix, Real) -> Matrix = { label, fx, x ->
  fx.map { if (it != 0.0) 1.0 else 0.0 }
  sum(fx)
  TODO("Verify")
}
