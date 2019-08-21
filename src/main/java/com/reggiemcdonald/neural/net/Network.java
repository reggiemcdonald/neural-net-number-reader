package com.reggiemcdonald.neural.net;

import java.util.*;

public class Network {
   private int layers;
   private int[] sizes;
   private List<Neuron> inputs, outputs;
   private List<List<Neuron>> hidden;

   public Network (int[] sizes) {
       if (sizes.length < 3)
           throw new RuntimeException("A network must have at least 3 layers");
       this.sizes   = sizes;
       this.layers  = sizes.length;
       this.inputs  = new ArrayList<>();
       this.outputs = new ArrayList<>();
       this.hidden  = new ArrayList<>();
       buildNetwork ();
   }

   private void buildNetwork () {
       // Build the input layer
       Random r = new Random ();
       for (int i = 0; i < sizes[0]; i++)
           inputs.add (new InputNeuron((float) r.nextGaussian()));
       // Build the hidden layer
       for (int i = 1; i < sizes.length-1; i++) {
           List<Neuron> neurons = new ArrayList<>();
           for (int j = 0; j < sizes[i]; j++) {
               neurons.add (new SigmoidNeuron());
           }
           hidden.add(neurons);
       }

       // Build the output layer
       for (int i = 0; i < sizes[sizes.length-1]; i++)
           outputs.add (new OutputNeuron());
       // Synapse inputs to first hidden
       createSynapses(inputs, hidden.get(0), r);
       // Synapse hidden layer
       for (int i = 0; i < hidden.size()-1; i++)
           createSynapses (hidden.get (i), hidden.get (i+1), r);
       createSynapses(hidden.get(hidden.size()-1), outputs, r);

   }

   private void createSynapses (List<Neuron> from, List<Neuron> to, Random r) {
       for (Neuron f : from) {
           for (Neuron t : to) {
               f.addSynapseFromThis(new Synapse(f,t, (float) r.nextGaussian()));
           }
       }
   }

   public List<Neuron> getInputs  () { return this.inputs; }
   public List<Neuron> getOutputs () { return this.outputs; }
   public Collection<Collection<Neuron>> getHiddens () { return Collections.unmodifiableCollection(this.hidden); }

}
