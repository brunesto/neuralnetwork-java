package bruno.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import bruno.nn.DataHelper.Stats;

/**
 * 
 * Bunch of helpers for training
 * 
 * @see TrainConfig
 */
public class TrainingHelper {

    /**
     * a sample
     */
    static class InAndOut {
        double[] input;
        double[] expected;

        public InAndOut(double[] input, double[] expected) {
            super();
            this.input = input;
            this.expected = expected;
        }

        @Override
        public String toString() {
            return "in:" + NeuralNet.s(input) + " expected:" + NeuralNet.s(expected);
        }

    }

    /**
     * errors
     */
    public static void layerErrorFunction(double[] outputs, double[] expecteds, double[] retVal) {
        for (int i = 0; i < expecteds.length; i++) {
            retVal[i] = NeuralNet.errorFunction(outputs[i], expecteds[i]);
        }
    }

    /**
     * error derivative
     */
    public static void layerErrorFunctiond(double[] outputs, double[] expecteds, double[] retVal) {
        for (int i = 0; i < expecteds.length; i++) {
            retVal[i] = NeuralNet.errorFunctiond(outputs[i], expecteds[i]);
        }
    }

    public static double[] computeError(double[] outputs, double[] expecteds) {
        double[] errors = new double[outputs.length];
        TrainingHelper.layerErrorFunction(outputs, expecteds, errors);
        double error = DataHelper.average(errors);
        //        double error = new Stats(errors).avg;
        int predicted = DataHelper.argmax(outputs);
        int expected = DataHelper.argmax(expecteds);
        int correct = (predicted == expected) ? 1 : 0;
        return new double[] { error, correct };
    }

    /**
     * compute error and accuracy over several  samples
     */
    public static double[] computeErrorAcc(List<InAndOut> samples, NeuralNet nn) {
        double errorAcc = 0;
        int correctAcc = 0;
        for (InAndOut sample : samples) {
            double[] pair = computeNetworkError(nn, sample);

            errorAcc += pair[0];
            correctAcc += pair[1];
        }
        double error = errorAcc / samples.size();
        double correct = correctAcc / (double) samples.size();
        if (NeuralNet.info)
            NeuralNet.log("error:" + error + " accuracy:" + correct);
        return new double[] { error, correct };
    }

    /**
     * compute error and accuracy over a single  samples
     */
    private static double[] computeNetworkError(NeuralNet nn, InAndOut sample) {
        if (NeuralNet.debug)
            NeuralNet.log("input " + NeuralNet.s(sample.input));
        double[] output = nn.computeFwd(sample.input);
        if (NeuralNet.debug) {
            NeuralNet.log("network output " + NeuralNet.s(output));
            NeuralNet.log("      expected " + NeuralNet.s(sample.expected));
        }
        double[] pair = computeError(output, sample.expected);
        if (NeuralNet.debug)
            NeuralNet.log("error:" + pair[0]);
        return pair;
    }

    public static class TrainConfig {
        /**
         * allows to select only a subset of the training data, so that the network can be updated after just a subset of samples
         */
        double reduceTrainingRatio = 1;
        /**
         * seed used to shuffle data
         */
        int seed;

        /**
         * the learning rate decays after each epoch
         */
        double rateDecay;
        /**
         * number of times the nn is trained against data in one epoch
         */
        int batches;

        int epochs;
    }

    /**
     * split the data in 2 parts, training and testing
     * @param data
     * @param ratio: how much data should be used for training, typical value: 0.9
     * @param nn
     * @param trainConfig
     */
    public static List<InAndOut> train(List<InAndOut> data, double trainingRatio, NeuralNet nn, TrainConfig trainConfig) {
        data = new ArrayList<TrainingHelper.InAndOut>(data);
        Random random = new Random(trainConfig.seed);
        Collections.shuffle(data, random);

        int splitIdx = (int) (trainingRatio * data.size());
        List<InAndOut> trainData = data.subList(0, splitIdx);
        List<InAndOut> testData = data.subList(splitIdx, data.size());

        train(trainData, testData, nn, trainConfig);
        return testData;
    }

    /**
     * runs a training session @see {@link TrainConfig}
     */
    public static void train(List<InAndOut> trainData, List<InAndOut> testData, NeuralNet nn, TrainConfig trainConfig) {

        trainData = new ArrayList<TrainingHelper.InAndOut>(trainData);
        //nn.normalizeWs();
        Random random = new Random(trainConfig.seed);
        int maxIdx = (int) Math.round(trainData.size() * trainConfig.reduceTrainingRatio);
        for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            if (maxIdx != trainData.size())
                Collections.shuffle(trainData, random);
            List<InAndOut> subset = trainData.subList(0, maxIdx);
            if (NeuralNet.info)
                NeuralNet.log("batches:" + trainConfig.batches);
            for (int batch = 0; batch < trainConfig.batches; batch++) {

                System.err.print(".");
                for (InAndOut sample : subset) {
                    if (NeuralNet.trace)
                        NeuralNet.log("sample:" + sample);
                    nn.computeFwdBwd(sample.input, sample.expected);

                }
                nn.applyDws();
                nn.resetDws();
                //nn.normalizeWs();

                if (NeuralNet.debug) {
                    for (int l = 0; l < nn.layers - 1; l++) {
                        NeuralNet.log("dw[" + l + "] stats:" + new Stats(nn.ws[0]));
                        NeuralNet.log("db[" + l + "] stats:" + new Stats(nn.bs[0]));

                    }
                }
            }
            nn.config.rate *= trainConfig.rateDecay;
            if (NeuralNet.info)
                NeuralNet.log("learning rate changed to " + nn.config.rate);
            computeErrorAcc(testData, nn);
            long endTime = System.currentTimeMillis();
            long deltaTime = endTime - startTime;
            if (NeuralNet.info)
                NeuralNet.log("epoch " + epoch + " done in " + deltaTime + " ");
        }

    }

}
