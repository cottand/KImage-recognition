import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.math.maximum
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.math.plus
import org.jetbrains.numkt.math.sum
import org.jetbrains.numkt.zeros
import kotlin.math.max

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

val svmLoss: LossFunc = { label, actual ->
  val margins = maximum(zeros(*actual.shape), actual - actual[label] + 1)
  margins[label] = 0.0
  sum(margins)
}

// TODO return type??
val dSvmLoss: LossFunc = { label, actual ->
  val margins = maximum(zeros(*actual.shape), actual - actual[label] + 1)
  margins.map { if (it != 0.0) 1.0 else 0.0 }
  sum(margins)
  TODO("Verify")
}

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

fun trainMNIST(initW: Matrix, training: List<Labelled>, validation: Set<Labelled>) {
  val batchSize = HyperParams.batchSize
  val batchNo = training.size / batchSize
  val batches = training.splitIntoBatches(batchSize)
  assert(batches.size in (batchNo - 1)..(batchNo + 1)) // TODO remove
  TODO()
}

