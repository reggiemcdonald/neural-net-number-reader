package com.reggiemcdonald.neural.net;

import java.util.ArrayList;
import java.util.List;

public class InputNeuron implements Neuron {

    private float outputtingSignal;
    private List<Synapse> sending;

    public InputNeuron (float outputtingSignal) {
        this.outputtingSignal = outputtingSignal;
        sending = new ArrayList<>();
    }

    @Override
    public float getOutputtingSignal() {
        return outputtingSignal;
    }

    @Override
    public Neuron setOutputtingSignal(float signal) {
        this.outputtingSignal = signal;
        return this;
    }

    @Override
    public void propagate() {
        for (Synapse s : sending) {
            s.setUpdated(true);
            s.getReceivingNeuron().propagate();
        }

    }

    @Override
    public float getBias() {
        return 1;
    }

    @Override
    public void setBias(float bias) {
        // Do Nothing
    }

    @Override
    public void addBiasUpdate(float biasUpdate) {
        // Do nothing
    }

    @Override
    public float getBiasUpdate () {
        return 0f;
    }

    @Override
    public void zeroBiasUpdate () {
        // Do Nothing
    }

    @Override
    public Neuron addSynapseOntoThis(Synapse s) {
        // TODO: Handle this better
        return null;
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        if (!sending.contains(s)) {
            sending.add(s);
            s.getReceivingNeuron().addSynapseOntoThis(s);
        }
        return this;
    }

    @Override
    public List<Synapse> getSynapsesOntoThis() {
        return null;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return null;
    }
}
