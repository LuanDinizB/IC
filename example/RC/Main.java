package RC;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class Samples {
    double[] features;
    int label;

    Samples(double[] features, int label) {
        this.features = features;
        this.label = label;
    }
}

class Perceptrons {
    private int numFeatures;
    private int numClasses;
    private double learningRate;
    private double[][] weights;
    private Random rand = new Random();

    public Perceptrons(int numFeatures, int numClasses, double learningRate) {
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.weights = new double[numClasses][numFeatures + 1]; // +1 para bias
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j <= numFeatures; j++) {
                weights[i][j] = -0.5 + rand.nextDouble(); // [-0.5, 0.5)
            }
        }
    }

    public void train(List<Samples> trainSet, int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int errors = 0;
            for (Samples s : trainSet) {
                double[] x = new double[s.features.length + 1];
                x[0] = 1.0; // bias
                System.arraycopy(s.features, 0, x, 1, s.features.length);

                int predicted = predict(x);
                if (predicted != s.label) {
                    errors++;
                    for (int j = 0; j < x.length; j++) {
                        weights[s.label][j] += learningRate * x[j];
                        weights[predicted][j] -= learningRate * x[j];
                    }
                }
            }
            double acc = 100.0 * (trainSet.size() - errors) / trainSet.size();
            System.out.printf("Época %d -> Erros: %d / %d | Acurácia treino: %.2f%%\n",
                    epoch, errors, trainSet.size(), acc);
        }
    }

    public double test(List<Samples> testSet) {
        int correct = 0;
        int errors = 0;

        for (Samples s : testSet) {
            double[] x = new double[s.features.length + 1];
            x[0] = 1.0;
            System.arraycopy(s.features, 0, x, 1, s.features.length);

            double[] scores = computeScores(x);
            int predicted = argMax(scores);
            double[] probs = softmax(scores);

            // imprimir informações detalhadas
            System.out.printf("Amostra: %s Esperado=%d Previsto=%d Probabilidades=%s\n",
                    Arrays.toString(s.features), s.label, predicted, Arrays.toString(probs));

            if (predicted == s.label) {
                correct++;
            } else {
                errors++;
            }
        }

        System.out.println("\nResultado Final:");
        System.out.println("Acertos: " + correct);
        System.out.println("Erros (amostras incorretas): " + errors);
        System.out.println("Soma total dos erros: " + errors);
        double acc = 100.0 * correct / testSet.size();
        System.out.printf("Taxa de Acerto: %.2f%%\n", acc);

        return acc;
    }

    private int predict(double[] x) {
        double[] scores = computeScores(x);
        return argMax(scores);
    }

    private double[] computeScores(double[] x) {
        double[] scores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            scores[c] = dot(weights[c], x);
        }
        return scores;
    }

    private int argMax(double[] scores) {
        int maxIndex = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    private double dot(double[] w, double[] x) {
        double sum = 0.0;
        for (int i = 0; i < w.length; i++) sum += w[i] * x[i];
        return sum;
    }

    private double[] softmax(double[] scores) {
        double max = Arrays.stream(scores).max().orElse(0.0);
        double sumExp = 0.0;
        double[] expScores = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            expScores[i] = Math.exp(scores[i] - max);
            sumExp += expScores[i];
        }
        double[] probs = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            probs[i] = expScores[i] / sumExp;
        }
        return probs;
    }
}

public class Main {

    public static List<Samples> loadIris(String path) throws IOException {
        List<Samples> dataset = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isBlank()) {
                    String[] parts = line.split(",");
                    double[] features = Arrays.stream(parts, 0, parts.length - 1)
                            .mapToDouble(Double::parseDouble).toArray();
                    int label = Integer.parseInt(parts[parts.length - 1]);
                    dataset.add(new Samples(features, label));
                }
            }
        }
        return dataset;
    }

    public static List<Samples> loadHorse(String path) throws IOException {
        List<Samples> dataset = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isBlank()) {
                    String[] parts = line.trim().split("\\s+");
                    double[] features = Arrays.stream(parts, 0, parts.length - 1)
                            .mapToDouble(v -> v.equals("?") ? 0.0 : Double.parseDouble(v))
                            .toArray();
                    int label = switch (parts[parts.length - 1]) {
                        case "1" -> 0;
                        case "2" -> 1;
                        default -> 0;
                    };
                    dataset.add(new Samples(features, label));
                }
            }
        }
        return dataset;
    }

    public static void main(String[] args) throws IOException {
        List<Samples> data = loadIris("example/RC/iris.csv");
        // List<Samples> data = loadHorse("example/horse-colic.data");

        Collections.shuffle(data, new Random(42));
        int trainSize = (int) (data.size() * 0.75);
        List<Samples> trainSet = data.subList(0, trainSize);
        List<Samples> testSet = data.subList(trainSize, data.size());

        int numFeatures = trainSet.get(0).features.length;
        int numClasses = (int) trainSet.stream().map(s -> s.label).distinct().count();

        Perceptrons perceptron = new Perceptrons(numFeatures, numClasses, 0.01);
        perceptron.train(trainSet, 30);

        perceptron.test(testSet);
    }
}
