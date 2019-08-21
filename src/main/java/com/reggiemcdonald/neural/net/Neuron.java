package com.reggiemcdonald.neural.net;

public interface Neuron {

    /**
     * @return the last outputted value of this
     */
    float getOutputtingSignal ();

    /**
     * Sets
     */
    Neuron setOutputtingSignal (float signal);


    /**
     * Updates the synapses this is incident on
     */
    void propagate ();

    /**
     * Returns the bias of this
     * @return
     */
    float getBias ();

    /**
     * Connects a neuron to this
     * @param s
     * @return this
     */
    Neuron addSynapseOntoThis (Synapse s);

    /**
     * Connects this to a neuron
     * @param s
     * @return this
     */
    Neuron addSynapseFromThis (Synapse s);


}
