package com.reggiemcdonald.neural.net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SigmoidNeuron implements Neuron, SigmoidShaped  {

    private float outputtingSignal;
    private List<Synapse> receiving, sending;
    private float bias;

    public SigmoidNeuron () {
        Random r  = new Random ();
        bias      = (float) r.nextGaussian();
        receiving = new ArrayList<> ();
        sending   = new ArrayList<> ();
    }
    @Override
    public float getOutputtingSignal () {
        return outputtingSignal;
    }

    @Override
    public Neuron setOutputtingSignal (float signal) {
        this.outputtingSignal = signal;
        return this;
    }

    @Override
    public void propagate () {
        if (!allHaveUpdated ()) return;
        // Integrate signals and set as output
        setOutputtingSignal (sigmoid());

        for (Synapse s : receiving)
            s.setUpdated(false);

        // Propogate down the network
        for (Synapse s : sending) {
            s.setUpdated(true);
            s.getReceivingNeuron().propagate();
        }
    }

    @Override
    public float getBias () {
        return bias;
    }

    @Override
    public Neuron addSynapseOntoThis (Synapse s) {
        if (!receiving.contains(s)) {
            receiving.add (s);
            s.getSendingNeuron().addSynapseFromThis(s);
        }
        return this;
    }

    @Override
    public Neuron addSynapseFromThis (Synapse s) {
        if (!sending.contains(s)) {
            sending.add (s);
            s.getReceivingNeuron().addSynapseOntoThis(s);
        }
        return this;
    }

    @Override
    public float sigmoid () {
        INDArray nd = Transforms.sigmoid (
                getWeights ()
                        .mmul(getInputs())
                        .add (bias)

        );
//        System.out.println(nd);
        return nd.getFloat(new int[]{0,0});
    }

    @Override
    public INDArray zed () {
        return getWeights().mmul(getInputs()).add (bias);
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
