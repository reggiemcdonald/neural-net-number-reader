package com.reggiemcdonald.neural.res;

public class NumberImage {
    public float[][] image;
    public float[] label;

    public NumberImage (float[][] image, int label) {
        this.image = image;
        this.label = new float[10];
        this.label[label] = 1f;
    }
}
