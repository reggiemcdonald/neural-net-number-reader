package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.NumberImage;

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

   public void input (float[][] input) {
       if (input.length == 0 || (input.length * input[0].length) != inputs.size())
           throw new RuntimeException("Invalid input");
       for (int x = 0; x < input.length; x++) {
           for (int y = 0; y < input[x].length; y++) {
               int index = (input[y].length * x) + y;
               inputs.get(index).setOutputtingSignal(input[x][y]).propagate();
           }
       }
   }

    /**
     * Perform stochastic gradient descent by means of backwards propagation
     * @param trainingData A list of NumberImage that contains the image and its true classification
     * @param epochs The number of rounds of training
     * @param batchSize The size of each training group
     * @param rate The learning rate
     * @param verbose true when progress should be printed to the console
     */
   public void learn (List<NumberImage> trainingData, int epochs, int batchSize, int rate, boolean verbose) {
       // TODO

       // Partition training data into random batches of batchSize
       Collections.shuffle(trainingData);
       List<List<NumberImage>> batches = new ArrayList<>();
       int idx = 0;
       for (int i = 0; i < trainingData.size() / batchSize; i++) {
           List<NumberImage> batch = new ArrayList<>(batchSize);
           for (int j = 0; j < batchSize; j++) {
               batch.add (trainingData.get (idx));
               idx++;
           }
           batches.add (batch);
       }

       // Perform small gradient descent epoch times
       // TODO
   }

    /**
     * Perform a SGD for a randomly partitioned training subset
     * @param batch a randomly partitioned set of trials
     */
   private void learn_batch (List<NumberImage> batch) {
       // TODO
       // For each of the tests in the batch
       // 1. Set the input layer
       // 2. Propagate
       // 3. Output S^L
       // 4. Backpropagate
   }


    /**
     * @return the index of the most activated neuron
     */
   public int getResult () {
       int idx = 0;
       float max_signal = outputs.get(0).getOutputtingSignal();
       for (int i = 0; i < outputs.size(); i++) {
           Neuron n = outputs.get(i);
           if (max_signal < n.getOutputtingSignal()) {
               idx = i;
               max_signal = n.getOutputtingSignal();
           }
       }
       return idx;
   }



}
