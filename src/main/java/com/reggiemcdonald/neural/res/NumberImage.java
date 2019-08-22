package com.reggiemcdonald.neural.res;

public class NumberImage {
    public float[][] image;
    public boolean[] label;

    public NumberImage (float[][] image, int label) {
        this.image = image;
        this.label = new boolean[10];
        this.label[label] = true;
    }
}
