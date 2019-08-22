package com.reggiemcdonald.neural.res;

public class NumberImage {
    public float[][] image;
    public int label;

    public NumberImage (float[][] image, int label) {
        this.image = image;
        this.label = label;
    }
}
