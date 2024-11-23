package bruno.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import bruno.nn.Helper.InAndOut;
import bruno.nn.Helper.TrainConfig;
import bruno.nn.NeuralNet.Config;

public class Mnist {
    static InAndOut readMnist12snn(ReadMnist r, int i) {
        double expecteds[] = Helper.categorize(9, r.labels[i]);
        double input[] = r.images.get(i);
        InAndOut inAndOut = new InAndOut(input, expecteds);
        return inAndOut;
    }

    static List<InAndOut> readMnistToData(String images, String labels) {
        List<InAndOut> mnistTrain = new ArrayList<InAndOut>();
        ReadMnist rTrain = new ReadMnist(images, labels);
        for (int i = 0; i < rTrain.labels.length; i++)
            mnistTrain.add(readMnist12snn(rTrain, i));
        return mnistTrain;
    }

    public static void main(String... args) throws IOException {




        //  computeErrorAcc(data, nn);

        //   train(data, 0.9, nn, 10, 0.90, 100);

        //        System.err.println(s(nn.ws));

        List<InAndOut> mnistTrain = readMnistToData("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/train-images-idx3-ubyte", "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/train-labels-idx1-ubyte");
        //        mnistTrain = mnistTrain.subList(0, 1000);
        List<InAndOut> mnistTest = readMnistToData("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-images-idx3-ubyte", "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-labels-idx1-ubyte");

        //        List<InAndOut> mnistTestSelect = new ArrayList<Helper.InAndOut>();
        //        for (InAndOut sample : mnistTest) {
        //            if (sample.expected[1] == 1 || sample.expected[6] == 1)
        //                mnistTestSelect.add(sample);
        //        }
        //        mnistTrain = mnistTrain.subList(0, 1000);
        mnistTest = mnistTest.subList(0, 1000);

        NeuralNet nn;
        NeuralNet.Config config;
        // no normalization ----------------------------------------------------

        config = new Config();
        config.layer_sizes = new int[] { 28 * 28, 512, 10 };
        config.rate = 0.1;
        nn = new NeuralNet(config);
        CsvHelper.consumeFile(nn, "/tmp/mnist-2.csv");

        //        CsvHelper.consumeFile(nn, "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/mnist-0.csv");
        //        CsvHelper.dumpNeuralNetToFile(nn, "/tmp/mnist-0.csv");

        // show initial error
        //        Helper.computeErrorAcc(mnistTrain, nn);
        SwingViewer.show(nn, mnistTest, 28, 16, 1);

        //        TrainConfig trainConfig = new TrainConfig();
        //        trainConfig.epochs = 50;
        //        trainConfig.rateDecay = 1.0;
        //        trainConfig.batches = 10;
        //        trainConfig.reduceTrainingRatio = 0.01;
        //        Helper.train(mnistTrain, mnistTest, nn, trainConfig);
        //        CsvHelper.dumpNeuralNetToFile(nn, "/tmp/mnist-3.csv");
        //        SwingViewer.show(nn, mnistTest, 28, 1);
    }



}
