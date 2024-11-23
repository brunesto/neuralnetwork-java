package bruno.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.DoubleFunction;

import bruno.nn.Helper.InAndOut;
import bruno.nn.Helper.TrainConfig;
import bruno.nn.NeuralNet.Config;

public class SimpleFunction {

    public static List<InAndOut> Sample(double from, double toIncl, double d, DoubleFunction<Double> f) {

        List<InAndOut> samples = new ArrayList<InAndOut>();
        for (double x = from; x <= toIncl; x += d) {
            double y = f.apply(x);
            samples.add(new InAndOut(new double[] { x }, new double[] { y }));
        }
        return samples;

    }

    public static void main(String... args) {

        List<InAndOut> data = Sample(-1, +1, 0.1, x -> x * x);


        NeuralNet.Config config;
        NeuralNet nn;

        //        NeuralNet.Config config = new Config();
        //        config.layer_sizes = new int[] { 1, 10, 1 };
        //        config.rate = 0.2;
        //        config.seed = 1;
        //        config.initial_weight_f = 1;
        //        NeuralNet nn = new NeuralNet(config);
        //        //System.err.println(NeuralNet.s(nn.ws));
        //        // Note that in this case since the output is not categorical, the accuracy is meaningless. it is always 1
        //        Helper.computeErrorAcc(data, nn);

        //nn.normalizeWs();
        //        System.err.println(NeuralNet.s(nn.ws));
        // Note that in this case since the output is not categorical, the accuracy is meaningless. it is always 1
        //Helper.computeErrorAcc(data, nn);

        config = new Config();
        config.layer_sizes = new int[] { 1, 5, 5, 1 };
        config.rate = 0.2;
        config.seed = 1;
        //        config.normalizeInitial = true;
        config.initial_weight_f = 2;
        nn = new NeuralNet(config);
        System.err.println(NeuralNet.s(nn.ws));


        //nn.normalizeWs();
        Helper.computeErrorAcc(data, nn);

        TrainConfig trainConfig = new TrainConfig();
        trainConfig.batches = 1000;
        trainConfig.epochs = 10;
        trainConfig.rateDecay = 0.95;
        Helper.train(data, 0.9, nn, trainConfig);
        Helper.computeErrorAcc(data, nn);

        SwingViewer.show(nn, data);
        //        // System.err.println(NeuralNet.s(nn.ws));
        //
        //

    }

}
