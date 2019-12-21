import org.jetbrains.numkt.array
import org.junit.jupiter.api.Test

class Test {
  @Test
  fun testArray() {
    /*
    1 2
    3 4
     */
    val w = array<Int>(listOf(listOf(1, 2), listOf(3, 4)))
    println(w[0..1])
  }
}