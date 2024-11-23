package bruno.nn;

import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.Supplier;

/**
 * a very simple neural network, the activation function is the same for all layers
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

        int layer_sizes[] = new int[] {};

        int seed = 0;
        boolean random_weight = true;
        double initial_weight_f = 0.1;
        double rate = 0.3;

        boolean normalizeInitial;

        public boolean normalizez = false;

        final double reluNegF = 0.1;

        public double sigma(double z) {
            if (z < 0)
                return z * reluNegF;
            else
                return z;
        }

        public double sigmad(double z) {
            if (z < 0)
                return reluNegF;
            else
                return 1;
        }
    }

    /**
     * return an array filled with 0
     */
    public static double[] zeros(int size) {
        return new double[size];
    }

    public static void zeros(double[] vs) {
        for (int i = 0; i < vs.length; i++)
            vs[i] = 0;
    }

    public static void zeros2d(double[][] vs) {
        for (int i = 0; i < vs.length; i++)
            zeros(vs[i]);
    }

    public static void fill(DoubleSupplier rand, double[] vs) {
        for (int i = 0; i < vs.length; i++)
            vs[i] = rand.getAsDouble();
    }

    public static void fill2d(DoubleSupplier rand, double[][] vs) {
        for (int i = 0; i < vs.length; i++)
            fill(rand, vs[i]);
    }

    /**
     * returns a 2d array filled with 0
     */
    public static double[][] zeros2d(int size1, int size2) {
        double[][] retVal = new double[size1][];
        for (int i = 0; i < size1; i++)
            retVal[i] = zeros(size2);

        return retVal;
    }

    public static double noNan(double x) {
        return x;
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
    double[][] ls;
    double[][] zs;
    double[][][] ws;
    double[][] bs;

    Config config;
    int layers;

    public NeuralNet(Config config) {
        this.config = config;

        layers = config.layer_sizes.length;

        this.ls = new double[layers][];
        for (int i = 0; i < layers; i++)
            this.ls[i] = zeros(config.layer_sizes[i]);

        // for each neuron, keep its z value (z: sum of weighted input (incl bias) before activation function)
        this.zs = new double[layers][];
        for (int i = 0; i < layers; i++)
            this.zs[i] = zeros(config.layer_sizes[i]);

        // weights
        // ws[1] are the weights to update l[1] from l[0]
        this.ws = new double[layers][][];
        for (int i = 1; i < layers; i++)
            this.ws[i - 1] = zeros2d(config.layer_sizes[i], config.layer_sizes[i - 1]);

        // biases
        this.bs = new double[layers][];
        for (int i = 1; i < layers; i++)
            this.bs[i - 1] = zeros(config.layer_sizes[i]);

        resetWBs();

        // Initialize derivative variables
        dh = 0;
        this.dls = new double[layers][];
        this.dls[0] = null; // we are not interested in the derivative of the cost over input layer
        for (int i = 1; i < this.layers; i++) {
            this.dls[i] = zeros(this.config.layer_sizes[i]);

        }

        this.dws = new double[layers][][];
        for (int li = 1; li < this.layers; li++)
            this.dws[li - 1] = zeros2d(this.config.layer_sizes[li], this.config.layer_sizes[li - 1]);

        this.dbs = new double[layers][];
        for (int li = 1; li < this.layers; li++)
            this.dbs[li - 1] = zeros(this.config.layer_sizes[li]);


    }

    public void reset() {
        resetDls();
        resetDws();
        resetWBs();
        resetLs();
    }

    int dh;
    double dls[][];
    //    double adws[];
    double dws[][][];
    //    double adbs[];
    double dbs[][];

    public void resetLs() {
        for (int i = 0; i < this.layers; i++)
            zeros(this.ls[i]);
        for (int i = 0; i < this.layers; i++)
            zeros(this.zs[i]);
    }

    public void resetWBs() {
        // now randomly populate weights
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

    public void resetDls() {
        // dls is the derivatives of cost over neurons, needs to be reset after each sample
            for (int i = 1; i < this.layers; i++)
                zeros(this.dls[i]);
    }

    public void resetDws() {
        // derivative of cost over weights
        // accumulated derivative
        // note that the derivative of the error shows the opposite direction of the gradient we want to follow
            for (int li = 1; li < this.layers; li++)
                zeros2d(this.dws[li - 1]);
            for (int li = 1; li < this.layers; li++)
                zeros(this.dbs[li - 1]);

            // dh is the number of samples
        this.dh = 0;
    }

    public double computeNeuronZ(double[] li, double[] wo, double b) {
        double z = 0;
        int ns = li.length;
        for (int ni = 0; ni < ns; ni++) {
            z += li[ni] * wo[ni];
        }
        z += b;

        return z;
    }

    /**
     * 
     * @param li input layer
     * @param wio weights between input and output layer
     * @param bo bias for output layer
     * @param lo output layer
     * @param zo z for neurons of output layer
     */
    public void computeLayer(double[] li, double[][] wio, double[] bo, double lo[], double[] zo, boolean normalizea) {
        double aaacc = 0;
        for (int no = 0; no < lo.length; no++) {
            double z = computeNeuronZ(li, wio[no], bo[no]);
            double a = config.sigma(z);
            lo[no] = a;
            zo[no] = z;
            aaacc += Math.abs(a);
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

    public double[] computeNetwork(double[] inputs) {
        // only reason for a system copy is for UI
        System.arraycopy(inputs, 0, ls[0], 0, inputs.length);

        for (int i = 1; i < layers; i++) {
            computeLayer(ls[i - 1], ws[i - 1], bs[i - 1], ls[i], zs[i], config.normalizez && i != layers - 1);
        }
        return ls[ls.length - 1];
    }

    // backward --------------------------------------

    public void updateBacktrack(double[] inputs, double[] expecteds) {

        computeNetwork(inputs);
        //double e = errorFunctionAcc(this.ls[layers], expecteds);

        resetDls();
        this.dh += 1;
        for (int j = 0; j < config.layer_sizes[layers - 1]; j++) {
            double dCostALj = errorFunctiond(this.ls[layers - 1][j], expecteds[j]);
            this.dls[layers - 1][j] += noNan(dCostALj);
        }

        for (int l = layers - 1; l > 0; l--) {
            for (int j = 0; j < this.ls[l].length; j++) {
                double dAZ = this.config.sigmad(this.zs[l][j]);
                double dCostA = this.dls[l][j];
                if (config.normalizez)
                    dCostA /= this.ls[l - 1].length;
                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    double dZW = this.ls[l - 1][k];

                    double dCostW = dCostA * dAZ * dZW;
                    this.dws[l - 1][j][k] += noNan(dCostW);
                }
                double dZB = 1;
                double dCostB = dCostA * dAZ * dZB;
                this.dbs[l - 1][j] += noNan(dCostB);

                if (l > 1) {
                    for (int k = 0; k < this.ls[l - 1].length; k++) {
                        double dZPrevA = this.ws[l - 1][j][k];
                        double dCostPrevA = (dZPrevA * dAZ * dCostA); // TODO dCostPrevA double overflow

                        this.dls[l - 1][k] += noNan(dCostPrevA);
                    }
                }
            }
        }
    }

    public void applyDws() {
        if (this.dh == 0)
            return;
        for (int l = 1; l < this.layers; l++) {
            for (int j = 0; j < this.ls[l].length; j++) {
                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    double gradientW = this.config.rate * this.dws[l - 1][j][k] / this.dh;
                    this.ws[l - 1][j][k] -= noNan(gradientW);
                }
                double gradientB = this.config.rate * this.dbs[l - 1][j] / this.dh;
                this.bs[l - 1][j] -= noNan(gradientB);
            }
        }
    }

    /**
     * normalize the weight so that all incoming weight + bias for a single neuron sum to 1 
     * (this step is necessary in java but not in python)
     */
    public void normalizeWs() {
        //        if (true)
        //            return;
        for (int l = 1; l < this.layers; l++) {

            double acc = 0;
            for (int j = 0; j < this.ls[l].length; j++) {

                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    acc += Math.abs(this.ws[l - 1][j][k]);
                }
                acc += Math.abs(this.bs[l - 1][j]);
            }
            double hits = this.ls[l].length * this.ls[l - 1].length;
            double invAcc = hits / acc;
            if (NeuralNet.trace)
                NeuralNet.log("l:" + l + " wacc:" + acc);
            for (int j = 0; j < this.ls[l].length; j++) {
                // normalize

                for (int k = 0; k < this.ls[l - 1].length; k++) {
                    this.ws[l - 1][j][k] *= invAcc;
                }
                this.bs[l - 1][j] *= invAcc;
            }
        }
    }

}
