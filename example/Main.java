import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.nio.file.Files.newBufferedReader;

public class Main {

        public static List<double[][]> lerArquivo(String caminhoArquivo, int numAmostras){

        List<double[][]> TREINO = new ArrayList<>();

        double[][] x = new double[numAmostras][27];
        double[][] y = new double[numAmostras][2];


        try (Scanner sc = new Scanner(new File(caminhoArquivo))) {

            for (int l = 0; l < numAmostras; l++) {
                for (int j = 0; j < 27; j++) {
                    x[l][j] = sc.nextDouble();
                }
                for (int j = 0; j < 2; j++) {
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

    private static void normalizarDados(double[][] data) {
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
            return lines
                    .map(line -> line + "&")
                    .collect(Collectors.toList());
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

//        shuffleFileLines("example/horse-colic-treino.data", "example/horse-colic-treino-atualizado.data");


        try {

            double[][][] AND = new double[][][] {
                    { { 0, 0 }, { 0 } },
                    { { 0, 1 }, { 0 } },
                    { { 1, 0 }, { 0 } },
                    { { 1, 1 }, { 1 } }
            };

            double[][][] XOR = new double[][][] {
                    { { 0, 0 }, { 0 } },
                    { { 0, 1 }, { 1 } },
                    { { 1, 0 }, { 1 } },
                    { { 1, 1 }, { 0 } }
            };

            double[][][] ROBO = new double[][][] {
                    { { 0, 0, 0 }, { 1, 1 } },
                    { { 0, 0, 1 }, { 0, 1 } },
                    { { 0, 1, 0 }, { 1, 0 } },
                    { { 0, 1, 1 }, { 0, 1 } },
                    { { 1, 0, 0 }, { 1, 0 } },
                    { { 1, 0, 1 }, { 1, 0 } },
                    { { 1, 1, 0 }, { 1, 0 } },
                    { { 1, 1, 1 }, { 1, 0 } },
            };


            List<double[][]> listaTeste = lerArquivo("example/horse-colic-teste2.data", 75);
            List<double[][]> listaTreino = lerArquivo("example/horse-colic-treino2.data", 225);

            double[][][] CAVALOTESTE = new double[listaTeste.size()][][];
            double[][][] CAVALOTREINO = new double[listaTreino.size()][][];

            for (int i = 0; i < listaTeste.size(); i++) {
                CAVALOTESTE[i]= listaTeste.get(i);
            }
            for (int i = 0; i < listaTreino.size(); i++) {
                CAVALOTREINO[i]= listaTreino.get(i);
            }


            int qtdEntradas = CAVALOTREINO[0][0].length;
            int qtdSaidas = CAVALOTREINO[0][1].length;

//            Perceptron perceptron = new Perceptron(qtdEntradas, qtdSaidas);
            MultiLayerPerceptron perceptron = new MultiLayerPerceptron(qtdEntradas, qtdSaidas, 0.003);
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
                    for (int k = 0; k < O.length; k++) {
                        erroAproximacaoAmostra += Math.abs(y_amostra[k] - O[k]);
                    }
                    erroAproximacaoEpoca += erroAproximacaoAmostra;

                    double maior = -1;

                    for (int z = 0; z < O.length; z++) {
                        if(O[z] > maior) maior = O[z];
                    }

                    double threshHoldValue = maior;

                    boolean amostraClassificadaErrada = false;
                    for (int k = 0; k < O.length; k++) {
                        double o_t = (O[k] >= threshHoldValue) ? 0.995 : 0.005;
                        if (Math.abs(y_amostra[k] - o_t) > 0) {
                            amostraClassificadaErrada = true;
                            break;
                        }
                    }
                    if (amostraClassificadaErrada) {
                        erroClassificacaoEpoca++;
                    }
                }
                for (int j = 0; j < CAVALOTESTE.length; j++) {
                    double[] x_amostra = CAVALOTESTE[j][0];
                    double[] y_amostra = CAVALOTESTE[j][1];
                    double[] OT = perceptron.testar(x_amostra, y_amostra);
                    double erroAproximacaoTesteAmostra = 0.0;
                    for (int k = 0; k < OT.length; k++) {
                        erroAproximacaoTesteAmostra += Math.abs(y_amostra[k] - OT[k]);
                    }
                    erroAproximacaoTesteEpoca += erroAproximacaoTesteAmostra;

                    double maior = -1;
                    for (int z = 0; z < OT.length; z++) {
                        if(OT[z] > maior) maior = OT[z];
                    }

                    double threshHoldValue = maior;

                    boolean amostraTesteClassificadaErrada = false;
                    for (int k = 0; k < OT.length; k++) {
                        double o_t = (OT[k] >= threshHoldValue) ? 0.995 : 0.005;
                        if (Math.abs(y_amostra[k] - o_t) > 0) {
                            amostraTesteClassificadaErrada = true;
                            break;
                        }
                    }
                    if (amostraTesteClassificadaErrada) {
                        erroClassificacaoTesteEpoca++;
                    }

                }
                System.out.printf("%d - %.4f - %.0f - %.4f - %.0f\n", (i + 1), erroAproximacaoEpoca, erroClassificacaoEpoca, erroAproximacaoTesteEpoca, erroClassificacaoTesteEpoca);
            }


        } catch (Exception e) {
            System.err.println("Ocorreu um erro inesperado durante a execução.");
            e.printStackTrace();
        }

    }
}