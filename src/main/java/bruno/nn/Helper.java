package bruno.nn;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import bruno.nn.SwingViewer.ArrayPanel.Stats;

public class Helper {

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

    public static double[] computeError(double[] outputs, double[] expecteds) {
        double[] errors = new double[outputs.length];
        Helper.layerErrorFunction(outputs, expecteds, errors);
        double error = Helper.average(errors);
        int predicted = argmax(outputs);
        int expected = argmax(expecteds);
        int correct = (predicted == expected) ? 1 : 0;
        return new double[] { error, correct };
    }

    public static double[] categorize(int max, int k) {
        double[] retVal = new double[max + 1];
        retVal[k] = 1;
        return retVal;
    }

    public static int argmax(double[] ds) {
        int imax = 0;
        double dmax = ds[0];
        for (int i = 1; i < ds.length; i++) {
            if (ds[i] > dmax) {
                dmax = ds[i];
                imax = i;
            }
        }
        return imax;
    }

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

    public static double noNan2(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x))
            throw new IllegalStateException();
        return x;
    }

    private static double[] computeNetworkError(NeuralNet nn, InAndOut sample) {
        if (NeuralNet.debug)
            NeuralNet.log("input " + NeuralNet.s(sample.input));
        double[] output = nn.computeNetwork(sample.input);
        if (NeuralNet.debug) {
            NeuralNet.log("network output " + NeuralNet.s(output));
            NeuralNet.log("      expected " + NeuralNet.s(sample.expected));
        }
        double[] pair = computeError(output, sample.expected);
        if (NeuralNet.debug)
            NeuralNet.log("error:" + pair[0]);
        return pair;
    }

    /**
     * split the data in 2 parts, training and testing
     * @param data
     * @param ratio: how much data should be used for training, typical value: 0.9
     * @param nn
     * @param trainConfig
     */
    public static void train(List<InAndOut> data, double trainingRatio, NeuralNet nn, TrainConfig trainConfig) {
        data = new ArrayList<Helper.InAndOut>(data);
        Random random = new Random(trainConfig.seed);
        Collections.shuffle(data, random);

        int splitIdx = (int) (trainingRatio * data.size());
        List<InAndOut> trainData = data.subList(0, splitIdx);
        List<InAndOut> testData = data.subList(splitIdx, data.size());

        train(trainData, testData, nn, trainConfig);
    }

    public static class TrainConfig {
        /**
         * allows to select only a subset of the training data, so that batches are faster
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
         * number of times the nn is train against data in one epoch
         */
        int batches;

        int epochs;
    }

    public static void train(List<InAndOut> trainData, List<InAndOut> testData, NeuralNet nn, TrainConfig trainConfig) {

        trainData = new ArrayList<Helper.InAndOut>(trainData);
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
                    nn.updateBacktrack(sample.input, sample.expected);

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

    /**
     * errors
     */
    public static void layerErrorFunction(double[] outputs, double[] expecteds, double[] retVal) {
        for (int i = 0; i < expecteds.length; i++) {
            retVal[i] = NeuralNet.errorFunction(outputs[i], expecteds[i]);
        }
    }

    /**
     * errors
     */
    public static void layerErrorFunctiond(double[] outputs, double[] expecteds, double[] retVal) {
        for (int i = 0; i < expecteds.length; i++) {
            retVal[i] = NeuralNet.errorFunctiond(outputs[i], expecteds[i]);
        }
    }

    public static double average(double[] vs) {
        double acc2 = 0;
        for (int i = 0; i < vs.length; i++) {
            acc2 += vs[i];
        }
        return acc2 / vs.length;
    }

    public static List<String[]> readCsv(String path) throws IOException {
        List<String[]> retVal = new ArrayList<>();
        for (String line : Files.readAllLines(Paths.get(path))) {
            if (line.startsWith("#") || line.isBlank())
                continue;
            String cols[] = line.split(",");
            //            InAndOut sample=get.apply(cols);
            //            if (sample!=null)
            retVal.add(cols);
        }
        return retVal;
    }

    public static double[] toArgmax(int size, int idx) {
        double[] retVal = new double[size];
        retVal[idx] = 1;
        return retVal;
    }

    public static Map<String, Integer> categorize(List<String[]> dataCsv) {
        Map<String, Integer> categories = new HashMap<String, Integer>();
        for (String[] row : dataCsv) {
            categories.putIfAbsent(row[4], categories.size());
        }
        return categories;
    }
}
