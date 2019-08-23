package com.reggiemcdonald.neural.net;

import java.util.List;

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

    void setBias (float bias);

    void addBiasUpdate (float biasUpdate);

    float getBiasUpdate ();

    void zeroBiasUpdate ();

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

    /**
     * @return the list of synapses onto this
     */
    List<Synapse> getSynapsesOntoThis ();

    /**
     * @return the list of synapses from this
     */
    List<Synapse> getSynapsesFromThis ();

}
