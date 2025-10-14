import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main2 {

    public static List<double[][]> lerArquivo(String caminhoArquivo, int numAmostras, int numLinha, int numValores) {

        List<double[][]> TREINO = new ArrayList<>();
        double[][] x = new double[numAmostras][numLinha];
        double[][] y = new double[numAmostras][numValores];

        try (Scanner sc = new Scanner(new File(caminhoArquivo))) {
            for (int l = 0; l < numAmostras; l++) {
                for (int j = 0; j < numLinha; j++) {
                    x[l][j] = sc.nextDouble();
                }
                for (int j = 0; j < numValores; j++) {
                    y[l][j] = sc.nextDouble();
                }
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + caminhoArquivo);
            e.printStackTrace();
        }

        normalizarDados(x);

        for (int i = 0; i < numAmostras; i++) {
            TREINO.add(new double[][]{x[i], y[i]});
        }
        return TREINO;
    }

    public static List<double[][]> lerArquivos(String caminhoArquivo, int numAmostras) {

        final int NUM_FEATURES = 10;
        final int NUM_CLASSES = 29;

        List<double[][]> treino = new ArrayList<>();
        double[][] x = new double[numAmostras][NUM_FEATURES];
        double[][] y = new double[numAmostras][NUM_CLASSES];

        try (Scanner sc = new Scanner(new File(caminhoArquivo))) {
            sc.useLocale(Locale.forLanguageTag("pt-BR"));

            if (sc.hasNextLine()) {
                sc.nextLine();
            }

            for (int l = 0; l < numAmostras; l++) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    if (sc.hasNextDouble()) {
                        x[l][j] = sc.nextDouble();
                    }
                }
                for (int j = 0; j < NUM_CLASSES; j++) {
                    if (sc.hasNextDouble()) {
                        y[l][j] = sc.nextDouble();
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO DE TREINO: " + caminhoArquivo);
            e.printStackTrace();
        }

        normalizarDados(x);

        for (int i = 0; i < numAmostras; i++) {
            treino.add(new double[][]{x[i], y[i]});
        }

        return treino;
    }

    public static void normalizarDados(double[][] data) {
        if (data.length == 0) return;
        int numColunas = data[0].length;
        for (int j = 0; j < numColunas; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (int i = 0; i < data.length; i++) {
                if (data[i][j] < min) min = data[i][j];
                if (data[i][j] > max) max = data[i][j];
            }
            double range = max - min;
            if (range == 0) range = 1;
            for (int i = 0; i < data.length; i++) {
                data[i][j] = (data[i][j] - min) / range;
            }
        }
    }

    public static List<String> read2(Path path, Charset cs) throws IOException {
        try (Stream<String> lines = Files.lines(path, cs)) {
            return lines.map(line -> line + "&").collect(Collectors.toList());
        }
    }

    public static void shuffleFileLines(String sourceFilePath, String destinationFilePath) throws IOException {
        Path sourcePath = Paths.get(sourceFilePath);
        Path destinationPath = Paths.get(destinationFilePath);
        List<String> lines = read2(sourcePath, StandardCharsets.UTF_8);
        Collections.shuffle(lines);
        Files.write(destinationPath, lines, StandardCharsets.UTF_8);
    }

    public static void main(String[] args) throws IOException {

        try {
            Scanner input = new Scanner(System.in);
            System.out.println("Selecione a base de dados:");
            System.out.println("1 - Horse Colic");
            System.out.println("2 - Abalone");
            System.out.print("Opção: ");
            int escolha = input.nextInt();

            double[][][] CAVALOTREINO = null;
            double[][][] CAVALOTESTE = null;
            MultiLayerPerceptron perceptron = null;

            if (escolha == 1) {
                System.out.println("Carregando base Horse Colic...");

                List<double[][]> listaTreino = lerArquivo("example/horse-colic-treino2.data", 225, 27, 2);
                List<double[][]> listaTeste = lerArquivo("example/horse-colic-teste2.data", 75, 27, 2);

                CAVALOTREINO = new double[listaTreino.size()][][];
                CAVALOTESTE = new double[listaTeste.size()][][];

                for (int i = 0; i < listaTreino.size(); i++) CAVALOTREINO[i] = listaTreino.get(i);
                for (int i = 0; i < listaTeste.size(); i++) CAVALOTESTE[i] = listaTeste.get(i);

                int qtdEntradas = CAVALOTREINO[0][0].length;
                int qtdSaidas = CAVALOTREINO[0][1].length;

                perceptron = new MultiLayerPerceptron(qtdEntradas, 10, qtdSaidas, 0.3);

            } else if (escolha == 2) {
                System.out.println("Carregando base Abalone...");

                List<double[][]> listaTreino = lerArquivos("example/abalone.train", 3139);
                List<double[][]> listaTeste = lerArquivos("example/abalone.test", 1048);


                CAVALOTREINO = new double[listaTreino.size()][][];
                CAVALOTESTE = new double[listaTeste.size()][][];

                for (int i = 0; i < listaTreino.size(); i++) CAVALOTREINO[i] = listaTreino.get(i);
                for (int i = 0; i < listaTeste.size(); i++) CAVALOTESTE[i] = listaTeste.get(i);

                int qtdEntradas = CAVALOTREINO[0][0].length;
                int qtdSaidas = CAVALOTREINO[0][1].length;

                // Inicializa o MLP
                perceptron = new MultiLayerPerceptron(qtdEntradas, 10, qtdSaidas, 0.01);
            } else {
                System.out.println("Opção inválida.");
                return;
            }

            int numEpocas = 100000;

            for (int i = 0; i < numEpocas; i++) {
                double erroAproximacaoEpoca = 0.0;
                double erroClassificacaoEpoca = 0.0;
                double erroAproximacaoTesteEpoca = 0.0;
                double erroClassificacaoTesteEpoca = 0.0;

                for (int j = 0; j < CAVALOTREINO.length; j++) {
                    double[] x_amostra = CAVALOTREINO[j][0];
                    double[] y_amostra = CAVALOTREINO[j][1];
                    double[] O = perceptron.treinar(x_amostra, y_amostra);

                    double erroAproximacaoAmostra = 0.0;
                    for (int k = 0; k < O.length; k++)
                        erroAproximacaoAmostra += Math.abs(y_amostra[k] - O[k]);
                    erroAproximacaoEpoca += erroAproximacaoAmostra;

                    double maior = -1;
                    for (double v : O) if (v > maior) maior = v;
                    double threshHoldValue = maior;

                    boolean amostraErrada = false;
                    for (int k = 0; k < O.length; k++) {
                        double o_t = (O[k] >= threshHoldValue) ? 0.995 : 0.005;
                        if (Math.abs(y_amostra[k] - o_t) > 0) {
                            amostraErrada = true;
                            break;
                        }
                    }
                    if (amostraErrada) erroClassificacaoEpoca++;
                }

                // Teste
                for (int j = 0; j < CAVALOTESTE.length; j++) {
                    double[] x_amostra = CAVALOTESTE[j][0];
                    double[] y_amostra = CAVALOTESTE[j][1];
                    double[] OT = perceptron.testar(x_amostra, y_amostra);

                    double erroAproximacaoTesteAmostra = 0.0;
                    for (int k = 0; k < OT.length; k++)
                        erroAproximacaoTesteAmostra += Math.abs(y_amostra[k] - OT[k]);
                    erroAproximacaoTesteEpoca += erroAproximacaoTesteAmostra;

                    double maior = -1;
                    for (double v : OT) if (v > maior) maior = v;
                    double threshHoldValue = maior;

                    boolean amostraTesteErrada = false;
                    for (int k = 0; k < OT.length; k++) {
                        double o_t = (OT[k] >= threshHoldValue) ? 0.995 : 0.005;
                        if (Math.abs(y_amostra[k] - o_t) > 0) {
                            amostraTesteErrada = true;
                            break;
                        }
                    }
                    if (amostraTesteErrada) erroClassificacaoTesteEpoca++;
                }

                System.out.printf("%d - %.4f - %.0f - %.4f - %.0f\n",
                        (i + 1), erroAproximacaoEpoca, erroClassificacaoEpoca,
                        erroAproximacaoTesteEpoca, erroClassificacaoTesteEpoca);
            }

        } catch (Exception e) {
            System.err.println("Ocorreu um erro inesperado durante a execução.");
            e.printStackTrace();
        }
    }
}
