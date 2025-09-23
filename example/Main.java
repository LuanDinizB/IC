import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

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

        int numAtributos = 28;
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

    public static void main(String[] args) {
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


            List<double[][]> lista = carregarDados("horse-colic.data");

            double[][][] CAVALO = new double[lista.size()][][];

            for (int i = 0; i < lista.size(); i++) {
                CAVALO[i]= lista.get(i);

            }

            double[][][] base = CAVALO;
            int qtdEntradas = base[0][0].length;
            int qtdSaidas = base[0][1].length;

//            Perceptron perceptron = new Perceptron(qtdEntradas, qtdSaidas);
            MultiLayerPerceptron perceptron = new MultiLayerPerceptron(qtdEntradas, qtdSaidas, 0.01);
            int numEpocas = 1000000;

            for (int i = 0; i < numEpocas; i++) {
                double erroAproximacaoEpoca = 0.0;
                double erroClassificacaoEpoca = 0.0;

                for (int j = 0; j < base.length; j++) {
                    double[] x_amostra = base[j][0];
                    double[] y_amostra = base[j][1];
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
                System.out.printf("%d - %.4f - %.0f\n", (i + 1), erroAproximacaoEpoca, erroClassificacaoEpoca);
            }

        } catch (Exception e) {
            System.err.println("Ocorreu um erro inesperado durante a execução.");
            e.printStackTrace();
        }
    }
}