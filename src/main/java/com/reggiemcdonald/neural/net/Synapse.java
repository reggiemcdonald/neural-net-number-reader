package com.reggiemcdonald.neural.net;

public class Synapse {
    private Neuron from, to;
    private float weight;
    private int index;
    private boolean updated;

    public Synapse (Neuron from, Neuron to, float weight) {
        this.from    = from;
        this.to      = to;
        this.weight  = weight;
        this.updated = false;
    }

    public Neuron getSendingNeuron () {
        return this.from;
    }


    public Neuron getReceivingNeuron () {
        return this.to;
    }

    public void setWeight (float weight) {
        this.weight = weight;
    }

    public float getWeight () {
        return this.weight;
    }

    public void setIndex (int index) {
        this.index = index;
    }

    public int getIndex () {
        return this.index;
    }

    public boolean isUpdated () { return this.updated; }

    public void setUpdated (boolean val) {
        this.updated = val;
    }
}
