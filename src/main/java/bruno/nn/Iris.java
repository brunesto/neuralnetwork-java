package bruno.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import bruno.nn.NeuralNet.Config;
import bruno.nn.TrainingHelper.InAndOut;
import bruno.nn.TrainingHelper.TrainConfig;

public class Iris {

    public static List<InAndOut> getIrisData() throws IOException {
        List<String[]> dataCsv = DataHelper.readCsv("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/iris.data");

        Map<String, Integer> categories = DataHelper.columnValueIndex(dataCsv, 4);

        List<InAndOut> data = new ArrayList<TrainingHelper.InAndOut>();
        for (String[] row : dataCsv) {
            data.add(new InAndOut(new double[] {
                    Double.parseDouble(row[0]), Double.parseDouble(row[1]), Double.parseDouble(row[2]), Double.parseDouble(row[3]) },
                    DataHelper.toArgmax(categories.size(), categories.get(row[4]))));
        }

        DataHelper.normalizeColumn(data, 0);
        DataHelper.normalizeColumn(data, 1);
        DataHelper.normalizeColumn(data, 2);
        DataHelper.normalizeColumn(data, 3);
        return data;

    }

    public static void main(String... args) throws IOException {

        List<InAndOut> data = getIrisData();


        NeuralNet.Config config;

        config = new Config();
        config.layer_sizes = new int[] { 4, 12, 3 };
        config.rate = 0.1;
        NeuralNet nn = new NeuralNet(config);

        // uncomment to load some precomputed weights
        //CsvHelper.consumeFile(nn, "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/weights.csv");

        // train the nn
        TrainConfig trainConfig = new TrainConfig();
        trainConfig.epochs = 100;
        trainConfig.rateDecay = 0.95;
        trainConfig.batches = 1000;
        trainConfig.reduceTrainingRatio = 0.2;

        // 90% of data will be for training, 10% for test 
        List<InAndOut> testData = TrainingHelper.train(data, 0.9, nn, trainConfig);

        // save weights
        // CsvHelper.dumpNeuralNetToFile(nn, "/tmp/iris-weights.csv");

        SwingViewer.show(nn, testData, 1, 1, 1);
    }

}
