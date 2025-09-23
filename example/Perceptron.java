// package com.example;

import java.util.Arrays;

public class Perceptron {
    private double[][] w;
    private int qtdIn, qtdOut;
    private double ni;

    public Perceptron(int qtdIn, int qtdOut) {
        ni = 0.0001;
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;

        w = new double[qtdIn + 1][qtdOut];

        for (int i = 0; i < w.length; i++) {
            Arrays.fill(this.w[i], -0.3);
        }
    }

    public double[] treinar(double[] xIn, double[] y) {
        double[] x = new double[xIn.length + 1];

        for (int i = 0; i < xIn.length; i++) {
            x[i] = xIn[i];
        }

        x[xIn.length] = 1;

        double[] O = new double[qtdOut];

        for (int j = 0; j < O.length; j++) {
            double u = 0.0;

            for (int i = 0; i < x.length; i++) {
                u += w[i][j] * x[i];
            }

            O[j] = 1 / (1 + Math.exp(-u));
        }

        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < x.length; i++) {
                this.w[i][j] += ni * (y[j] - O[j]) * x[i];
            }
        }

        return O;
    }
}