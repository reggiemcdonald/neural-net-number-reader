package com.reggiemcdonald.neural.net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class OutputNeuron implements Neuron, SigmoidShaped {
    List<Synapse> receiving;
    float outputtingSignal, bias;

    public OutputNeuron () {
        Random r = new Random();
        receiving = new ArrayList<> ();
        outputtingSignal = (float) r.nextGaussian();
        bias = (float) r.nextGaussian();
    }

    @Override
    public float getOutputtingSignal() {
        return outputtingSignal;
    }

    @Override
    public Neuron setOutputtingSignal(float signal) {
        outputtingSignal = signal;
        return this;
    }

    @Override
    public float sigmoid() {
        INDArray nd = Transforms.sigmoid (
                getWeights ()
                        .mmul(getInputs())
                        .add (bias)

        );
        return nd.getFloat(new int[]{0,0});
    }

    @Override
    public INDArray zed () {
        return getWeights().mmul(getInputs()).add (bias);
    }

    @Override
    public void propagate() {
        if (!allHaveUpdated()) return;

        setOutputtingSignal(sigmoid());

        for (Synapse s : receiving)
            s.setUpdated(false);
    }

    @Override
    public float getBias() {
        return 0;
    }

    @Override
    public Neuron addSynapseOntoThis(Synapse s) {
        if (!receiving.contains(s)) {
            receiving.add (s);
            s.getSendingNeuron().addSynapseFromThis(s);
        }
        return this;
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        return this;
    }

    // Create a row vector of weights
    private INDArray getWeights () {
        float[] arr = new float[receiving.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = receiving.get(i).getWeight();
        }
        return Nd4j.create(arr);
    }

    // Create a column vector of inputs
    private INDArray getInputs () {
        float[] arr = new float[receiving.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = receiving.get(i)
                    .getSendingNeuron()
                    .getOutputtingSignal();
        }
        return Nd4j.create (arr, new int[] {arr.length,1});
    }

    private boolean allHaveUpdated () {
        boolean hasUpdated = true;
        for (Synapse s : receiving)
            hasUpdated =  hasUpdated && s.isUpdated();
        return hasUpdated;
    }
}
