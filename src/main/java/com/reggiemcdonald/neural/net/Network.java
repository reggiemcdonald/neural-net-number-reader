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
               inputs.get(index).setOutputtingSignal(input[x][y] / 255).propagate();
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
       System.out.println("Before training");
       test (trainingData);
       // Perform small gradient descent epoch times
       int count = 1;
       for (int i = 1; i <= epochs; i++) {
           System.out.println("Beginning epoch " + i);
//           int bnum = 1;
           for (List<NumberImage> batch : batches) {
//               System.out.println("Beginning batch " + bnum + " out of " + batches.size());
               learn_batch(batch, eta);
//               bnum++;
           }
//           test (trainingData);
       }
       System.out.println("After training");
       test (trainingData);
//       print (inputs);
//       for (List<Neuron> layer : hidden)
//           print (layer);
//       print (outputs);
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
       INDArray output_activation   = getOutputINDArray();
       INDArray delta               = deltaL (output_activation, expected_activation);
       // 2. Backpropagate to earlier layers by setting bias updates and weight updates
       // Start with the output layer
       INDArray layer_weights = getLayerWeights (outputs);
       INDArray layer_activation, input_activations;
       input_activations = getLayerActivationArray (hidden.get (hidden.size () - 1));
       setLayerBiasUpdate (outputs, delta);
       setLayerWeightUpdate (outputs, delta, input_activations);


       // Radiate towards the input layer
       for (int i = hidden.size()-1; i > -1; i--) {
           List<Neuron> layer = hidden.get(i);
           layer_activation = getLayerActivationArray (layer);
           input_activations = getLayerActivationArray(i == 0 ? inputs : hidden.get (i-1));
           delta            = updateDelta (delta, layer_activation, layer_weights);
           layer_weights    = getLayerWeights (layer);
           setLayerBiasUpdate   (layer, delta);
           setLayerWeightUpdate (layer, delta, input_activations);
       }
   }

   private void setLayerBiasUpdate (List<Neuron> layer, INDArray delta) {
       for (int i = 0; i < layer.size(); i++)
           layer.get(i).addBiasUpdate(delta.getFloat(i,1));
   }

   private void setLayerWeightUpdate (List<Neuron> layer, INDArray delta, INDArray input_activations) {
       INDArray weightUpdate     = delta.mmul(input_activations.transpose());
       int neuron_num = 0;
       for (Neuron n : layer) {
           List<Synapse> synapses = n.getSynapsesOntoThis();
           for (int i = 0; i < synapses.size(); i++)
               synapses.get(i).addWeightUpdate(weightUpdate.getFloat(neuron_num,i));
           neuron_num++;
       }
   }

   private INDArray updateDelta (INDArray old_delta, INDArray layer_activation, INDArray layer_weights) {
        INDArray sigmoid_prime = Transforms.sigmoidDerivative (layer_activation);
        return layer_weights.transpose().mmul (old_delta).mul  (sigmoid_prime);
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
   public int getResult (float[] arr) {
       int idx = 0;
       float max_signal = Float.MIN_VALUE;
       for (int i = 0; i < arr.length; i++) {
           if (arr[i] > max_signal) {
               idx = i;
               max_signal = arr[i];
           }
       }
       return idx;
   }

   public float[] getOutput () {
       float [] o = new float[outputs.size()];
       int i = 0;
       for (Neuron n : outputs) {
           o[i] = n.getOutputtingSignal();
           i++;
       }
       return o;
   }

   public INDArray getOutputINDArray() {
       float [] o = getOutput();
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

   private INDArray getLayerWeights (List<Neuron> layer) {
       List<INDArray> weights = new ArrayList<>(layer.size());
       int rows = 0;
       for (Neuron n : layer) {
           List<Synapse> synapses = n.getSynapsesOntoThis();
           INDArray weight = getWeights (synapses);
           rows = weight.rows();
           weights.add(weight);
       }
       return Nd4j.create (weights, new int[]{layer.size(),rows});
   }

    private INDArray getWeights (List<Synapse> synapses) {
        INDArray arr = Nd4j.create (synapses.size(), 1);
        for (int i = 0; i < synapses.size(); i++)
            arr.put (i, 1, synapses.get(i).getWeight());
        return arr;
    }

    public void print (List<Neuron> layer) {
        System.out.println("===== Layer =====");
        for (Neuron n : layer) {
            System.out.println("Neuron bias: " + n.getBias());
            List<Synapse> synapses = n.getSynapsesFromThis();
            for (Synapse s : synapses)
                System.out.println("     Synapse weight: " + s.getWeight());
        }
        System.out.println("===== END Layer =====");
    }

    public void test (List<NumberImage> data) {
       int correct = 0, idx = 0, out_of = data.size();
        System.out.println("Testing...");
       for (NumberImage x : data) {
//           System.out.println("Testing " + idx);
           input (x.image);
           if (getResult(getOutput()) == getResult(x.label))
               correct++;
//           idx++;
       }
        System.out.println(correct + " correct, out of " + out_of);
    }

}
