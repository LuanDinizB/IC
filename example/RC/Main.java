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

class RCClassifier {
    private int numClasses;
    private int numFeatures;
    private double[][] medias;
    private double[] priors;

    public RCClassifier(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.medias = new double[numClasses][numFeatures];
        this.priors = new double[numClasses];
    }

    public void train(List<Samples> trainSet) {
        int[] classCounts = new int[numClasses];

        for (Samples s : trainSet) {
            for (int j = 0; j < numFeatures; j++) {
                medias[s.label][j] += s.features[j];
            }
            classCounts[s.label]++;
        }

        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] > 0) {
                for (int j = 0; j < numFeatures; j++) {
                    medias[c][j] /= classCounts[c];
                }
            }
        }

        for (int c = 0; c < numClasses; c++) {
            priors[c] = (double) classCounts[c] / trainSet.size();
        }

        System.out.println("Modelo treinado. Matriz de médias:");
        for (int c = 0; c < numClasses; c++) {
            System.out.println("Classe " + c + ": " + Arrays.toString(medias[c]));
        }
        System.out.println("\nProbabilidades a priori: " + Arrays.toString(priors));
    }

    public double test(List<Samples> testSet) {
        int correct = 0;
        for (Samples s : testSet) {
            int predicted = classify(s.features);
            if (predicted == s.label) {
                correct++;
            }
            System.out.printf("Amostra: %s Esperado=%d Previsto=%d\n",
                    Arrays.toString(s.features), s.label, predicted);
        }

        double accuracy = 100.0 * correct / testSet.size();
        System.out.println("\nResultado Final");
        System.out.println("Acertos: " + correct);
        System.out.println("Total de amostras de teste: " + testSet.size());
        System.out.printf("Taxa de Acerto: %.2f%%\n", accuracy);
        return accuracy;
    }

    private int classify(double[] features) {
        double Cp = -Double.MAX_VALUE;
        for (double feature : features) {
            if (feature > Cp) {
                Cp = feature;
            }
        }

        double[] scores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            scores[c] = Cp * priors[c];
        }

        int maxIndex = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
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
                    double[] features = new double[21];
                    for(int i = 0; i < 21; i++) {
                        features[i] = parts[i+2].equals("?") ? 0.0 : Double.parseDouble(parts[i+2]);
                    }
                    int label = parts[22].equals("1") ? 0 : 1;
                    dataset.add(new Samples(features, label));
                }
            }
        }
        return dataset;
    }

    public static void main(String[] args) throws IOException {
        //List<Samples> data = loadIris("example/RC/iris.csv");
        List<Samples> data = loadHorse("example/horse-colic.data");

        Collections.shuffle(data, new Random(42));
        int trainSize = (int) (data.size() * 0.75);
        List<Samples> trainSet = data.subList(0, trainSize);
        List<Samples> testSet = data.subList(trainSize, data.size());

        if (trainSet.isEmpty()) {
            System.out.println("Erro: O conjunto de treino está vazio.");
            return;
        }

        int numFeatures = trainSet.get(0).features.length;
        Set<Integer> uniqueLabels = new HashSet<>();
        for (Samples s : trainSet) {
            uniqueLabels.add(s.label);
        }
        int numClasses = uniqueLabels.size();

        RCClassifier classifier = new RCClassifier(numClasses, numFeatures);
        classifier.train(trainSet);
        System.out.println("\n--- Iniciando Teste ---");
        classifier.test(testSet);
    }
}