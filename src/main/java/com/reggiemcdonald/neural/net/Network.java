package com.reggiemcdonald.neural.net;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class Network {
   private int layers;
   private int[] sizes;
   private List<Neuron> inputs, outputs;

   public Network (int[] sizes) {
       this.sizes  = sizes;
       this.layers = sizes.length;
       buildNetwork ();
   }

   private void buildNetwork () {
       // TODO Stub
   }

   public Collection<Neuron> getInputs  () { return Collections.unmodifiableCollection(this.inputs); }
   public Collection<Neuron> getOutputs () { return Collections.unmodifiableCollection(this.outputs); }

}
