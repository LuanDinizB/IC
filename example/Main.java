import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void separateHorseData(String sourceFilePath) {
        String outputFile1 = "outcome1.data";
        String outputFile2 = "outcome2.data";

        // O bloco try-with-resources garante que os leitores e escritores sejam fechados automaticamente.
        try (BufferedReader reader = new BufferedReader(new FileReader(sourceFilePath));
             BufferedWriter writer1 = new BufferedWriter(new FileWriter(outputFile1));
             BufferedWriter writer2 = new BufferedWriter(new FileWriter(outputFile2))) {

            String line;
            while ((line = reader.readLine()) != null) {
                // Remove espaços em branco no início/fim para uma verificação confiável.
                String trimmedLine = line.trim();

                if (trimmedLine.endsWith("1")) {
                    writer1.write(line);
                    writer1.write("&");
                    writer1.newLine(); // Adiciona uma nova linha para manter a formatação.
                } else if (trimmedLine.endsWith("2")) {
                    writer2.write(line);
                    writer2.write("&");
                    writer2.newLine(); // Adiciona uma nova linha.
                }
            }

            System.out.println("Arquivos separados com sucesso!");
            System.out.println("Dados com final '1' foram salvos em: " + outputFile1);
            System.out.println("Dados com final '2' foram salvos em: " + outputFile2);

        } catch (IOException e) {
            System.err.println("Ocorreu um erro ao processar os arquivos: " + e.getMessage());
            e.printStackTrace();
        }
    }

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


    public static List<double[][]> carregarDados(String caminhoArquivo) throws IOException {
        List<double[][]> baseDeDados = new ArrayList<>();
        StringBuilder conteudoArquivo = new StringBuilder();

        try (BufferedReader br = new BufferedReader(new FileReader(caminhoArquivo))) {
            String linha;
            while ((linha = br.readLine()) != null) {
                conteudoArquivo.append(linha).append(" ");
            }
        }
        String conteudoLimpo = conteudoArquivo.toString();
        for (int i = 1; i < 500; i++) {
            conteudoLimpo = conteudoLimpo.replace("", "");
        }

        conteudoLimpo = conteudoLimpo.replace("?", "0");
        String[] tokens = conteudoLimpo.trim().split("\\s+");

        int numAtributos = 27;
        if (tokens.length < numAtributos) {
            throw new IOException("Arquivo de dados parece estar vazio ou mal formatado após a limpeza.");
        }
        int numAmostras = tokens.length / numAtributos;

        double[][] dadosBrutos = new double[numAmostras][numAtributos];
        for (int i = 0; i < numAmostras; i++) {
            for (int j = 0; j < numAtributos; j++) {
                dadosBrutos[i][j] = Double.parseDouble(tokens[i * numAtributos + j]);
            }
        }

        double[][] x = new double[numAmostras][numAtributos - 2];
        double[][] y = new double[numAmostras][1];

        for (int i = 0; i < numAmostras; i++) {
            int x_col_index = 0;
            for (int j = 0; j < numAtributos; j++) {
                if (j == 2) continue;
                if (j == 22) {
                    y[i][0] = (dadosBrutos[i][j] == 1.0) ? 0.0 : 1.0;
                } else {
                    x[i][x_col_index] = dadosBrutos[i][j];
                    x_col_index++;
                }
            }
        }

        normalizarDados(x);

        for (int i = 0; i < numAmostras; i++) {
            baseDeDados.add(new double[][]{x[i], y[i]});
        }

        return baseDeDados;
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

    public static void shuffleFileLines(String sourceFilePath, String destinationFilePath) throws IOException {
        Path sourcePath = Paths.get(sourceFilePath);
        Path destinationPath = Paths.get(destinationFilePath);
        List<String> lines = Files.readAllLines(sourcePath, StandardCharsets.UTF_8);
        Collections.shuffle(lines);
        Files.write(destinationPath, lines, StandardCharsets.UTF_8);
    }


    public static void main(String[] args) throws IOException {

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


            List<double[][]> listaTeste = lerArquivo("example/horse-colic-teste.data", 75);
            List<double[][]> listaTreino = lerArquivo("example/horse-colic-treino.data", 225);

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
            MultiLayerPerceptron perceptron = new MultiLayerPerceptron(qtdEntradas, qtdSaidas, 0.001);
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

                    boolean amostraClassificadaErrada = false;
                    for (int k = 0; k < O.length; k++) {
                        double o_t = (O[k] >= 0.5) ? 1.0 : 0.0;
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

                    boolean amostraTesteClassificadaErrada = false;
                    for (int k = 0; k < OT.length; k++) {
                        double o_t = (OT[k] >= 0.5) ? 1.0 : 0.0;
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