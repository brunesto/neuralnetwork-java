package bruno.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import bruno.nn.Helper.InAndOut;
import bruno.nn.Helper.TrainConfig;
import bruno.nn.NeuralNet.Config;

public class Iris {

    public static void main(String... args) throws IOException {

        //  computeErrorAcc(data, nn);

        //   train(data, 0.9, nn, 10, 0.90, 100);

        //        System.err.println(s(nn.ws));

        List<String[]> dataCsv = Helper.readCsv("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/iris.data");

        Map<String, Integer> categories = Helper.categorize(dataCsv);

        List<InAndOut> data = new ArrayList<Helper.InAndOut>();
        for (String[] row : dataCsv) {
            //System.err.print("" + Arrays.asList(row));
            data.add(new InAndOut(new double[] {
                    Double.parseDouble(row[0]), Double.parseDouble(row[1]), Double.parseDouble(row[2]), Double.parseDouble(row[3]) },
                    Helper.toArgmax(categories.size(), categories.get(row[4]))));
        }

        normalizeColumn(data, 0);
        normalizeColumn(data, 1);
        normalizeColumn(data, 2);
        normalizeColumn(data, 3);

        NeuralNet nn;
        NeuralNet.Config config;
        // no normalization ----------------------------------------------------

        config = new Config();
        config.layer_sizes = new int[] { 4, 12, 3 };
        config.rate = 0.1;
        nn = new NeuralNet(config);

        CsvHelper.consumeFile(nn, "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/weights.csv");

        TrainConfig trainConfig = new TrainConfig();
        trainConfig.epochs = 10;
        trainConfig.rateDecay = 0.95;
        trainConfig.batches = 1000;
        //trainConfig.reduceTrainingRatio = 0.2;
        //Helper.train(data, 0.9, nn, trainConfig);

        SwingViewer.show(nn, data, 1, 1, 1);
    }

    private static void normalizeColumn(List<InAndOut> data, int c) {
        double min = data.get(0).input[c];
        double max = data.get(0).input[c];
        for (int i = 0; i < data.size(); i++) {
            double v = data.get(i).input[c];
            if (v > max)
                max = v;
            if (v < min)
                min = v;
        }
        System.err.println("col " + c + " min:" + min + " max:" + max);
        double range = max - min;

        for (int i = 0; i < data.size(); i++) {
            double v = data.get(i).input[c];
            v = (v - min) / range;
            data.get(i).input[c] = v;
        }
    }

}
