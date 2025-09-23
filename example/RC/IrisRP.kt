package RC

import java.io.File
import kotlin.random.Random

data class Sample(val features: DoubleArray, val label: Int)

class Perceptron(
    private val numFeatures: Int,
    private val numClasses: Int,
    private val learningRate: Double = 0.1
) {
    private val weights: Array<DoubleArray> =
        Array(numClasses) { DoubleArray(numFeatures + 1) { Random.nextDouble(-0.5, 0.5) } } // +1 bias

    fun train(trainSet: List<Sample>, epochs: Int) {
        for (epoch in 1..epochs) {
            var errors = 0
            for (s in trainSet) {
                val x = doubleArrayOf(1.0, *s.features) // adiciona bias na frente
                val predicted = predict(x)
                if (predicted != s.label) {
                    errors++
                    // regra de atualização (One-vs-Rest)
                    for (j in x.indices) {
                        weights[s.label][j] += learningRate * x[j]   // reforça classe correta
                        weights[predicted][j] -= learningRate * x[j] // penaliza classe errada
                    }
                }
            }
            val acc = 100.0 * (trainSet.size - errors) / trainSet.size
            println("Época $epoch -> Erros: $errors / ${trainSet.size} | Acurácia treino: %.2f%%".format(acc))
        }
    }

    fun test(testSet: List<Sample>): Double {
        var correct = 0
        for (s in testSet) {
            val x = doubleArrayOf(1.0, *s.features)
            val predicted = predict(x)
            if (predicted == s.label) correct++
        }
        return 100.0 * correct / testSet.size
    }

    private fun predict(x: DoubleArray): Int {
        val scores = DoubleArray(numClasses) { c -> dot(weights[c], x) }
        return scores.indices.maxByOrNull { scores[it] } ?: 0
    }

    private fun dot(w: DoubleArray, x: DoubleArray): Double {
        var sum = 0.0
        for (i in w.indices) sum += w[i] * x[i]
        return sum
    }
}

fun loadIris(path: String): List<Sample> {
    return File(path).readLines()
        .filter { it.isNotBlank() }
        .map { line ->
            val parts = line.split(",")
            val features = parts.dropLast(1).map { it.toDouble() }.toDoubleArray()
            val label = parts.last().toInt() // assumindo labels 0,1,2
            Sample(features, label)
        }
}

fun loadHorse(path: String): List<Sample> {
    return File(path).readLines()
        .filter { it.isNotBlank() }
        .map { line ->
            val parts = line.trim().split(Regex("\\s+"))
            val features = parts.dropLast(1).map { v -> if (v == "?") 0.0 else v.toDouble() }.toDoubleArray()
            val label = when (parts.last()) {
                "1" -> 0 // sobreviveu
                "2" -> 1 // morreu
                else -> 0
            }
            Sample(features, label)
        }
}

fun main() {
    // Escolha aqui sua base:
   //  val data = loadIris("example/RC/iris.csv")
    val data = loadHorse("example/horse-colic.data")


    val path = "example/RC/iris.csv"  // <-- coloque o CSV na mesma pasta
    // val data = loadDataset(path).shuffled(Random(42))

    val shuffled = data.shuffled(Random(42))
    val trainSize = (shuffled.size * 0.75).toInt()
    val trainSet = shuffled.take(trainSize)
    val testSet = shuffled.drop(trainSize)

    val numFeatures = trainSet[0].features.size
    val numClasses = trainSet.map { it.label }.toSet().size

    val perceptron = Perceptron(numFeatures, numClasses, learningRate = 0.01)
    perceptron.train(trainSet, epochs = 30)

    val accTest = perceptron.test(testSet)
    println("\nAcurácia no teste final: %.2f%%".format(accTest))
}
