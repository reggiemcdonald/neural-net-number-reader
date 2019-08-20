package com.reggiemcdonald.neural.net;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Neuron {

    /**
     * @return the last outputted value of this
     */
    float getOutput ();

    /**
     * Propogate the signal across the network
     * @param inputs: An INDArray of inputs
     */
    void  propogate (INDArray inputs);
}
