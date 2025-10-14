import java.io.*;
import java.util.*;

public class AbaloneLoader {

    /**
     * Lê a base Abalone no formato compatível com a função lerArquivo da Main.
     * Cada linha gera uma amostra com X[8] e Y[1].
     */
    public static List<double[][]> lerAbalone(String caminhoArquivo, int numAmostras) {
        List<double[][]> DADOS = new ArrayList<>();

        double[][] x = new double[numAmostras][8];
        double[][] y = new double[numAmostras][1];

        try (BufferedReader br = new BufferedReader(new FileReader(caminhoArquivo))) {
            String linha;
            int linhaIndex = 0;

            while ((linha = br.readLine()) != null && linhaIndex < numAmostras) {
                linha = linha.trim();
                if (linha.isEmpty()) continue;

                String[] partes = linha.split(",");

                if (partes.length != 9) continue;

                // Conversão do atributo nominal 'Sex' para número
                double sexo;
                switch (partes[0].trim()) {
                    case "M" -> sexo = 1.0;
                    case "F" -> sexo = 0.0;
                    case "I" -> sexo = 0.5;
                    default -> sexo = 0.0;
                }

                x[linhaIndex][0] = sexo;

                // Lê as demais features numéricas
                for (int j = 1; j < 8; j++) {
                    try {
                        x[linhaIndex][j] = Double.parseDouble(partes[j]);
                    } catch (NumberFormatException e) {
                        x[linhaIndex][j] = 0.0;
                    }
                }

                // Saída (rings)
                try {
                    y[linhaIndex][0] = Double.parseDouble(partes[8]);
                } catch (NumberFormatException e) {
                    y[linhaIndex][0] = 0.0;
                }

                linhaIndex++;
            }
        } catch (IOException e) {
            System.err.println("ERRO AO LER ARQUIVO ABALONE: " + caminhoArquivo);
            e.printStackTrace();
        }

        // Usa a mesma normalização da Main
        Main.normalizarDados(x);
        Main.normalizarDados(y);

        // Constrói a lista compatível com o formato Horse Colic
        for (int i = 0; i < numAmostras; i++) {
            DADOS.add(new double[][]{x[i], y[i]});
        }

        return DADOS;
    }
}
