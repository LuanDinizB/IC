package RC;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Iris {

    // Classe para armazenar uma amostra
    static class Sample {
        double[] features;
        int label;

        Sample(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    public static void main(String[] args) throws IOException {
        // ============================
        // 1. Carregar a base Iris (CSV)
        // ============================
        // Espera-se um CSV no formato:
        // sepal_length,sepal_width,petal_length,petal_width,class
        // Ex: 5.1,3.5,1.4,0.2,0
        String path = "example/RC/iris.csv"; // <-- coloque o caminho do dataset
        List<Sample> data = loadDataset(path);

        // separar treino (75%) e teste (25%)
        Collections.shuffle(data, new Random(42));
        int trainSize = (int) (data.size() * 0.75);

        List<Sample> trainSet = data.subList(0, trainSize);
        List<Sample> testSet = data.subList(trainSize, data.size());

        int numClasses = 3;
        int numFeatures = 4;

        // ============================
        // 2. Criar o "modelo"
        // ============================
        double[][] matriz = new double[numClasses][numFeatures];
        int[] classCounts = new int[numClasses];

        // somar atributos por classe
        for (Sample s : trainSet) {
            for (int j = 0; j < numFeatures; j++) {
                matriz[s.label][j] += s.features[j];
            }
            classCounts[s.label]++;
        }

        // calcular médias
        for (int c = 0; c < numClasses; c++) {
            for (int j = 0; j < numFeatures; j++) {
                matriz[c][j] /= classCounts[c];
            }
        }

        // calcular probabilidades a priori
        double[] priors = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            priors[c] = (double) classCounts[c] / trainSet.size();
        }

        // ============================
        // 3. Testar e classificar
        // ============================
        int correct = 0;
        for (Sample s : testSet) {
            int predicted = classify(s.features, matriz, priors);
            if (predicted == s.label) {
                correct++;
            }
        }

        double accuracy = (double) correct / testSet.size() * 100.0;

        // ============================
        // 4. Resultados
        // ============================
        System.out.println("Matriz de médias (modelo): ");
        for (int c = 0; c < numClasses; c++) {
            System.out.println(Arrays.toString(matriz[c]));
        }

        System.out.println("\nProbabilidades a priori: " + Arrays.toString(priors));
        System.out.printf("\nAcurácia no teste: %.2f%%\n", accuracy);
    }

    // Função para classificar uma amostra
    static int classify(double[] sample, double[][] matriz, double[] priors) {
        int numClasses = priors.length;
        // Cp = max(x1, x2, ..., xn)
        double Cp = Arrays.stream(sample).max().getAsDouble();

        double[] probs = new double[numClasses];
        double denom = 0.0;

        for (int c = 0; c < numClasses; c++) {
            probs[c] = Cp * priors[c];
            denom += probs[c];
        }

        // normalizar
        for (int c = 0; c < numClasses; c++) {
            probs[c] = (denom > 0) ? probs[c] / denom : 0.0;
        }

        // retorna índice da maior probabilidade
        int maxIndex = 0;
        for (int c = 1; c < numClasses; c++) {
            if (probs[c] > probs[maxIndex]) {
                maxIndex = c;
            }
        }
        return maxIndex;
    }

    // Função para carregar dataset CSV
    static List<Sample> loadDataset(String path) throws IOException {
        List<Sample> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            String[] parts = line.split(",");
            double[] features = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                features[i] = Double.parseDouble(parts[i]);
            }
            int label = Integer.parseInt(parts[parts.length - 1]);
            data.add(new Sample(features, label));
        }
        br.close();
        return data;
    }
}
