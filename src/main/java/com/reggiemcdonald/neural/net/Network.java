package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.NumberImage;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

public class Network {
   private int layers;
   private int[] sizes;
   private List<Neuron> inputs;
   private List<Neuron> outputs;
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
     * @param eta The learning rate
     * @param verbose true when progress should be printed to the console
     */
   public void learn (List<NumberImage> trainingData, int epochs, int batchSize, float eta, boolean verbose) {
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
       for (List<NumberImage> batch : batches)
           learn_batch (batch, eta);
       // TODO
   }

    /**
     * Perform a SGD for a randomly partitioned training subset
     * @param batch a randomly partitioned set of trials
     */
   private void learn_batch (List<NumberImage> batch, float eta) {
       // TODO
       // For each of the tests in the batch
       for (NumberImage x : batch) {
           // 1. Set the input layer and propagate
           input (x.image);
           // 2. Backpropagate
           backPropagate (x);
       }
       // 3. Update the weights and biases
       finalizeLearning (outputs, eta, batch.size());
       for (List<Neuron> layer : hidden)
           finalizeLearning (layer, eta, batch.size());
   }

   private void backPropagate (NumberImage x) {
       // 1. Compute the error in the output layer
       INDArray expected_activation = Nd4j.create (x.label, new int[] {x.label.length, 1});
       INDArray output_activation   = getOutput ();
       INDArray delta               = deltaL (output_activation, expected_activation);
       // 2. Backpropagate to earlier layers by setting bias updates and weight updates
       setLayerBiasUpdate (outputs, delta);
       setLayerWeightUpdate (outputs, delta);
       for (List<Neuron> layer : hidden) {
           delta = updateDelta  (layer, delta);
           setLayerBiasUpdate   (layer, delta);
           setLayerWeightUpdate (layer, delta);
       }
   }

   private void setLayerBiasUpdate (List<Neuron> layer, INDArray bias) {
       // TODO
   }

   private void setLayerWeightUpdate (List<Neuron> layer, INDArray bias) {
       // TODO
   }

   private INDArray updateDelta (List<Neuron> layer, INDArray old_delta) {
       return old_delta; // TODO stub
   }

   private void finalizeLearning (List<Neuron> layer, float eta, int batchSize) {
       for (Neuron n : layer) {
           n.setBias(n.getBias() - (eta/batchSize * n.getBiasUpdate()));
           for (Synapse s : n.getSynapsesOntoThis()) {
               s.setWeight(s.getWeight() - (eta/batchSize * s.getWeightUpdate()));
               s.zeroWeightUpdate();
           }
           n.zeroBiasUpdate();
       }
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

   public INDArray getOutput () {
       float [] o = new float[outputs.size()];
       int i = 0;
       for (Neuron n : outputs) {
           o[i] = n.getOutputtingSignal();
           i++;
       }
       return Nd4j.create (o, new int[] {o.length,1});
   }

   private INDArray deltaL (INDArray output_activation, INDArray expected_activation) {
       // delta^L = (a^L - y) Hadamard Prod. sigmoid_prime (z^L)
       // where z^L is a
       INDArray cost_derivative = getCostDerivative (output_activation, expected_activation); // (a^L - y)
       INDArray zL              = getZedL ();
       return cost_derivative.mul (
               Transforms
                       .sigmoidDerivative (zL)
                       .castTo (DataType.FLOAT)
       );

   }

    /**
     * Compute the error
     * @param output_activation
     * @param expected_activation
     * @return
     */
   private INDArray getCostDerivative (INDArray output_activation, INDArray expected_activation) {
       return output_activation.sub (expected_activation);
   }

   private INDArray getZedL () {
       List<INDArray> zeds = new ArrayList<>();
       for (Neuron n : outputs) {
           SigmoidShaped s = (SigmoidShaped) n;
           zeds.add(s.zed());
       }
       return Nd4j.create (zeds, new int[]{zeds.size(),1});
   }

    /**
     * @param layer a list of disjoint neurons representing a single layer
     * @return a vector of neuronal activations
     */
   private INDArray getLayerActivationArray (List<Neuron> layer) {
       INDArray arr = Nd4j.create (layer.size(), 1);
       for (int i = 0; i < layer.size(); i++)
           arr.put (i, 1, layer.get(i).getOutputtingSignal());
       return arr;
   }




}
