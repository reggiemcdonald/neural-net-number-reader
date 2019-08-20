package com.reggiemcdonald.neural;

import com.reggiemcdonald.neural.net.Network;

/**
 * Hello world!
 *
 */
public class App {
    public static void main( String[] args )
    {
        Network n = new Network (new int[]{2,3,1});
        System.out.println(n.getBiases());
        System.out.println("Weight:");
        System.out.println(n.getWeights().get(1));
    }
}
