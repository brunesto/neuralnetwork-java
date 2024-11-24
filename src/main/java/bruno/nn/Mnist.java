package bruno.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import bruno.nn.NeuralNet.Config;
import bruno.nn.TrainingHelper.InAndOut;
import bruno.nn.TrainingHelper.TrainConfig;

public class Mnist {

    public static List<InAndOut> readMnistToData(String images, String labels) {
        List<InAndOut> mnistTrain = new ArrayList<InAndOut>();
        ReadMnist rTrain = new ReadMnist(images, labels);
        for (int i = 0; i < rTrain.labels.length; i++) {
            double expecteds[] = DataHelper.toArgmax(10, rTrain.labels[i]);
            double input[] = rTrain.images.get(i);
            InAndOut inAndOut = new InAndOut(input, expecteds);
            mnistTrain.add(inAndOut);
        }
        return mnistTrain;
    }

    public static void main(String... args) throws IOException {

        List<InAndOut> mnistTrain = readMnistToData("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/train-images-idx3-ubyte", "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/train-labels-idx1-ubyte");
        List<InAndOut> mnistTest = readMnistToData("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-images-idx3-ubyte", "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-labels-idx1-ubyte");

        mnistTest = mnistTest.subList(0, 1000);

        NeuralNet.Config config;

        config = new Config();
        config.layer_sizes = new int[] { 28 * 28, 512, 10 };
        config.rate = 0.1;
        NeuralNet nn = new NeuralNet(config);


        CsvHelper.consumeFile(nn, "/tmp/mnist-2.csv");

        // show initial error
        // Helper.computeErrorAcc(mnistTest, nn);

        // SwingViewer.show(nn, mnistTest, 28, 16, 1);

        TrainConfig trainConfig = new TrainConfig();
        trainConfig.epochs = 10;
        trainConfig.rateDecay = 1.0;
        trainConfig.batches = 10;
        trainConfig.reduceTrainingRatio = 0.01;
        TrainingHelper.train(mnistTrain, mnistTest, nn, trainConfig);
        CsvHelper.dumpNeuralNetToFile(nn, "/tmp/mnist-3.csv");
        //        SwingViewer.show(nn, mnistTest, 28, 1);
    }

}
