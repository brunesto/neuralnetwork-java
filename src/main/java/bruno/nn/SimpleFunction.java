package bruno.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.DoubleFunction;

import bruno.nn.TrainingHelper.InAndOut;
import bruno.nn.TrainingHelper.TrainConfig;
import bruno.nn.NeuralNet.Config;

/**
 *  Note that in this case since the output is not categorical, the accuracy is meaningless. it is always 1
 */
public class SimpleFunction {

    public static List<InAndOut> getSamples(double from, double toIncl, double d, DoubleFunction<Double> f) {

        List<InAndOut> samples = new ArrayList<InAndOut>();
        for (double x = from; x <= toIncl; x += d) {
            double y = f.apply(x);
            samples.add(new InAndOut(new double[] { x }, new double[] { y }));
        }
        return samples;

    }

    public static void main(String... args) {

        List<InAndOut> data = getSamples(-1, +1, 0.1, x -> x * x);


        NeuralNet.Config config;



        config = new Config();
        config.layer_sizes = new int[] { 1, 5, 5, 1 };
        config.rate = 0.2;
        config.seed = 1;
        config.initial_weight_f = 2;

        NeuralNet nn = new NeuralNet(config);
        System.err.println(NeuralNet.s(nn.ws));

        TrainingHelper.computeErrorAcc(data, nn);

        TrainConfig trainConfig = new TrainConfig();
        trainConfig.batches = 1000;
        trainConfig.epochs = 10;
        trainConfig.rateDecay = 0.95;
        TrainingHelper.train(data, 0.9, nn, trainConfig);

        TrainingHelper.computeErrorAcc(data, nn);

        SwingViewer.show(nn, data);

    }

}
