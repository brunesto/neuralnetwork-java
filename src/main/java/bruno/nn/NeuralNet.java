package bruno.nn;

import java.util.Random;

/**
 * A very simple self-contained neural network, the activation function is the same for all layers
 * 
 * It can learn MNIST in under <10 minutes with 95% accuracy
 * 
 */
public class NeuralNet {

    static long start = System.currentTimeMillis();

    final static boolean info = true;
    final static boolean debug = false;
    final static boolean trace = false;

    static void log(String msg) {
        System.err.println(String.format("%08d", System.currentTimeMillis() - start) + ":" + msg);
    }

    static class Config {

        /**
         * defines the number of neurons per layer
         */
        int layer_sizes[] = new int[] {};

        /**
         * seed used for random generator
         */
        int seed = 0;

        /**
         * Use a low initial weight avoid gradient explosion
         */
        double initial_weight_f = 0.1;

        /**
         * learning rate
         */
        double rate = 0.3;

        // TODO: remove me
        boolean random_weight = true;
        boolean normalizeInitial;
        //        public boolean normalizez = false;

        //-- activation function ------------------------------------
        /**
         * factor of the activation function for z<0 (for z>0 this factor is always 1.0)
         */
        final double reluNegF = 0.1;

        /**
         * activation function
         * @param z sum of weighted values of neurons from previous layer + bias
         */
        public double sigma(double z) {
            if (z < 0)
                return z * reluNegF;
            else
                return z;
        }

        /**
         * derivative of activation function
         */
        public double sigmad(double z) {
            if (z < 0)
                return reluNegF;
            else
                return 1;
        }
    }

    /**
     * error function for a single neuron. aka cost - low cost is good
     */
    public static double errorFunction(double output1, double expected1) {
        double diff = (output1 - expected1);
        double r = diff * diff;
        return r;
    }

    /**
     * derivative of errorFunction()
     */
    public static double errorFunctiond(double output1, double expected1) {
        double diff = (output1 - expected1);
        double r = 2 * diff;
        return r;
    }

    public static String s(double x) {
        if (x >= 0)
            return " " + String.format("%6.4f", x);
        else
            return String.format("%6.4f", x);
    }

    public static String s(double[] x) {
        if (x == null)
            return null;
        String s = "[";
        for (int i = 0; i < x.length; i++) {
            if (i != 0)
                s += ",";
            s += s(x[i]);
        }
        s += "]";
        return s;
    }

    public static String s(double[][] x) {
        if (x == null)
            return null;

        String s = "[";
        for (int i = 0; i < x.length; i++) {
            s += "\n";
            s += s(x[i]);
        }
        s += "]";
        return s;
    }

    public static String s(double[][][] x) {
        if (x == null)
            return null;

        String s = "[";
        for (int i = 0; i < x.length; i++) {
            s += "\n\n\n";
            s += s(x[i]);
        }
        s += "]";
        return s;
    }

    // activations

    /**
     * values of z for neurons before the activation function
     */
    double[][] zs;

    /**
     * layers, i.e. neurons activation
     */
    double[][] ls;

    /**
     * weight between neurons (each neuron of a given layer is connected to all neurons of previous layer)
     * ws[0] are the weights to update neurons of l[1] based on  l[0]
     */
    double[][][] ws;
    /**
     * bias, one per neuron
     */
    double[][] bs;

    // derivatives

    /**
     * number of forward passes
     */
    int dh;
    /**
     * network cost derivative vs derivative of: neuron activations for the last fwd pass
     */
    double dls[][];
    /**
     * accumulated network cost derivative vs derivative of: weights between neurons   
     */
    double dws[][][];
    /**
     * accumulated network cost derivative vs derivative of: bias   
     */
    double dbs[][];

    /**
     * network definition
     */
    Config config;

    /**
     * number of layers, for conveniency 
     */
    int layers;

    //-- initialization + resets ------------------------------------------------------------------------------------------------
    public NeuralNet(Config config) {
        this.config = config;

        layers = config.layer_sizes.length;

        this.ls = new double[layers][];
        for (int i = 0; i < layers; i++)
            this.ls[i] = DataHelper.zeros(config.layer_sizes[i]);

        // for each neuron, keep its z value (z: sum of weighted input (incl bias) before activation function)
        this.zs = new double[layers][];
        for (int i = 0; i < layers; i++)
            this.zs[i] = DataHelper.zeros(config.layer_sizes[i]);

        // weights
        // ws[0] are the weights used to compute l[1] from l[0]
        this.ws = new double[layers][][];
        for (int i = 1; i < layers; i++)
            this.ws[i - 1] = DataHelper.zeros2d(config.layer_sizes[i], config.layer_sizes[i - 1]);

        // biases, bs[0] will be used to compute l[1]
        this.bs = new double[layers][];
        for (int i = 1; i < layers; i++)
            this.bs[i - 1] = DataHelper.zeros(config.layer_sizes[i]);

        // populate weights
        resetWBs();

        // Initialize derivative variables
        dh = 0;
        this.dls = new double[layers][];
        this.dls[0] = null; // we are not interested in the derivative of the cost over input layer
        for (int i = 1; i < this.layers; i++) {
            this.dls[i] = DataHelper.zeros(this.config.layer_sizes[i]);

        }

        this.dws = new double[layers][][];
        for (int li = 1; li < this.layers; li++)
            this.dws[li - 1] = DataHelper.zeros2d(this.config.layer_sizes[li], this.config.layer_sizes[li - 1]);

        this.dbs = new double[layers][];
        for (int li = 1; li < this.layers; li++)
            this.dbs[li - 1] = DataHelper.zeros(this.config.layer_sizes[li]);

    }

    /**
     * reset the whole nn
     */
    public void reset() {
        resetDls();
        resetDws();
        resetWBs();
        resetLs();
    }

    /**
     * reset the neuron activation values (a+z)
     */
    public void resetLs() {
        for (int i = 0; i < this.layers; i++)
            DataHelper.zeros(this.ls[i]);
        for (int i = 0; i < this.layers; i++)
            DataHelper.zeros(this.zs[i]);
    }

    /**
     * randomly populate weights - typically happens only once
     */
    public void resetWBs() {

        Random random = new Random(config.seed);
        for (int l = 1; l < layers; l++) {
            for (int no = 0; no < ws[l - 1].length; no++) {
                double normalizef = (config.normalizeInitial ? (1 + ws[l - 1][no].length) : 1);
                for (int ni = 0; ni < ws[l - 1][no].length; ni++) {
                    double w;
                    w = (random.nextDouble() - 0.5);
                    ws[l - 1][no][ni] = w * config.initial_weight_f * normalizef;
                }
                double w;
                w = (random.nextDouble() - 0.5);
                bs[l - 1][no] = w * config.initial_weight_f * normalizef;
            }
        }
    }

    /**
     * Reset the derivatives of activation 
     * Notes:
     * -This should not be necessary as these values would be overriden during every backtrack - but it is easier to debug + to see in the UI  
     * -Derivative of z is not kept
     */
    public void resetDls() {
        // dls is the derivatives of cost over neurons, needs to be reset after each sample
        for (int i = 1; i < this.layers; i++)
            DataHelper.zeros(this.dls[i]);
    }

    /**
     * Reset the accumulated derivatives weights - must be done before training on a batch of sample
     */
    public void resetDws() {
        // derivative of cost over weights
        // accumulated derivative
        // note that the derivative of the error shows the opposite direction of the gradient we want to follow
        for (int li = 1; li < this.layers; li++)
            DataHelper.zeros2d(this.dws[li - 1]);
        for (int li = 1; li < this.layers; li++)
            DataHelper.zeros(this.dbs[li - 1]);

        // dh is the number of samples
        this.dh = 0;
    }

    // -- fwd -----------------------------------------------------------------------------------
    /**
     * compute z: the weighted sum of previous layer activations + bias
     * @param li: input layer (i.e. previous)
     * @param wo: weights
     * @param b: bias
     */
    public double computeZ(double[] li, double[] wo, double b) {
        double z = 0;
        int ns = li.length;
        for (int ni = 0; ni < ns; ni++) {
            z += li[ni] * wo[ni];
        }
        z += b;

        return z;
    }

    /**
     * compute all the activations of neurons for a given layer
     * @param li input layer
     * @param wio weights between input and given layer
     * @param bo bias 
     * @param lo used to store the activation of neurons of given layer
     * @param zo used to store z for neurons of given layer
     */
    public void computeLayer(double[] li, double[][] wio, double[] bo, double lo[], double[] zo) {
        //        double aaacc = 0;

        for (int no = 0; no < lo.length; no++) {
            double z = computeZ(li, wio[no], bo[no]);
            double a = config.sigma(z);
            lo[no] = a;
            zo[no] = z;
            //            aaacc += Math.abs(a);
        }
        // System.err.println("before:" + new Stats(lo));
        //        if (normalizea) {
        //            double f = 4 * lo.length / aaacc; crap..aaacc.
        //            //            System.err.println("f:" + f);
        //            for (int no = 0; no < lo.length; no++)
        //                lo[no] *= f;
        //        }
        //   System.err.println("after:" + new Stats(lo));

    }

    /**
     * compute the full network in fwd pass
     */
    public double[] computeFwd(double[] inputs) {
        // only reason for a system copy is for UI
        System.arraycopy(inputs, 0, ls[0], 0, inputs.length);

        for (int i = 1; i < layers; i++) {
            computeLayer(ls[i - 1], ws[i - 1], bs[i - 1], ls[i], zs[i]);
        }
        return ls[ls.length - 1];
    }

    //-- backtracking --------------------------------------

    public void computeFwdBwd(double[] inputs, double[] expecteds) {

        computeFwd(inputs);

        resetDls();
        // keep count of samples, because the derivatives of weight and biases are beeing accumulated
        this.dh += 1;

        // output layer derivative depends on the error function
        for (int j = 0; j < config.layer_sizes[layers - 1]; j++) {
            double dCostALj = errorFunctiond(this.ls[layers - 1][j], expecteds[j]);
            this.dls[layers - 1][j] += DataHelper.noNan(dCostALj);
        }

        for (int l = layers - 1; l > 0; l--) {
            for (int j = 0; j < this.ls[l].length; j++) {
                // the derivative of network cost VS a specific weight is computed from 3 parts 

                // 1) derivative of network cost VS the activation the neuron computed from this weight amongst others
                double dCostA = this.dls[l][j];
                // 2) derivative of the activation function vs z (sum of weighted activation of previous layer ' neurons + bias)
                double dAZ = this.config.sigmad(this.zs[l][j]);

                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    // 3) derivative of z vs the weight 
                    double dZW = this.ls[l - 1][k];

                    double dCostW = dCostA * dAZ * dZW;
                    this.dws[l - 1][j][k] += DataHelper.noNan(dCostW);
                }
                // for bias, the formula is the same, and the derivative of z VS the bias is always 1
                double dZB = 1;
                double dCostB = dCostA * dAZ * dZB;
                this.dbs[l - 1][j] += DataHelper.noNan(dCostB);

                // now compute the derivative of network cost VS neuron activations of this layer, because it will be used for the adjacent previous layer 
                if (l > 1) {
                    for (int k = 0; k < this.ls[l - 1].length; k++) {
                        double dZPrevA = this.ws[l - 1][j][k];
                        double dCostPrevA = (dCostA * dAZ * dZPrevA);
                        this.dls[l - 1][k] += DataHelper.noNan(dCostPrevA);
                    }
                }
            }
        }
    }

    /**
     * apply the partial derivatives of cost vs all weights and biases
     * Note that since we want to improve (lower) the score, we need
     * to apply this changes in opposite direction
     */
    public void applyDws() {
        if (this.dh == 0)
            return;
        for (int l = 1; l < this.layers; l++) {
            for (int j = 0; j < this.ls[l].length; j++) {
                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    // The accumulated derivative (dws) is divided by number of samples (dh) to get the average derivative
                    double gradientW = this.dws[l - 1][j][k] / this.dh;
                    // apply the learning rate
                    gradientW *= this.config.rate;
                    // apply the gradient in opposite direction
                    this.ws[l - 1][j][k] -= DataHelper.noNan(gradientW);
                }
                // bias follow the same logic as weights
                double gradientB = this.config.rate * this.dbs[l - 1][j] / this.dh;
                this.bs[l - 1][j] -= DataHelper.noNan(gradientB);
            }
        }
    }

    //    /**
    //     * TODO: remove me -
    //     * normalize the weight so that all incoming weight + bias for a single neuron sum to 1 
    //     * (this step is necessary in java but not in python)
    //     */
    //    public void normalizeWs() {
    //        //        if (true)
    //        //            return;
    //        for (int l = 1; l < this.layers; l++) {
    //
    //            double acc = 0;
    //            for (int j = 0; j < this.ls[l].length; j++) {
    //
    //                for (int k = 0; k < this.ls[l - 1].length; k++) {
    //                    acc += Math.abs(this.ws[l - 1][j][k]);
    //                }
    //                acc += Math.abs(this.bs[l - 1][j]);
    //            }
    //            double hits = this.ls[l].length * this.ls[l - 1].length;
    //            double invAcc = hits / acc;
    //            if (NeuralNet.trace)
    //                NeuralNet.log("l:" + l + " wacc:" + acc);
    //            for (int j = 0; j < this.ls[l].length; j++) {
    //                // normalize
    //
    //                for (int k = 0; k < this.ls[l - 1].length; k++) {
    //                    this.ws[l - 1][j][k] *= invAcc;
    //                }
    //                this.bs[l - 1][j] *= invAcc;
    //            }
    //        }
    //    }

}
