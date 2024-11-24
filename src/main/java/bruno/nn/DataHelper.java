package bruno.nn;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;

import bruno.nn.TrainingHelper.InAndOut;

public class DataHelper {

    static class Stats {
        double max;
        double min;
        double acc;
        double avg;
        double aavg;
        double aacc;
        int n;

        void init(double v0) {
            aacc = 0;
            max = v0;
            min = v0;
        }

        Stats(double[] vs) {
            init(vs[0]);

            process(vs);

            done();
        }

        Stats(double[][] vss) {
            init(vss[0][0]);

            for (int i = 0; i < vss.length; i++) {
                double[] vs = vss[i];
                process(vs);
            }
            done();
        }

        private void process(double[] vs) {
            for (int i = 0; i < vs.length; i++) {
                double v = vs[i];
                process(v);
            }
        }

        private void done() {
            avg = acc / n;
            aavg = aacc / n;
        }

        private void process(double v) {
            if (v < min)
                min = v;
            if (v > max)
                max = v;
            acc += v;
            aacc += Math.abs(v);
            n++;
        }

        @Override
        public String toString() {
            String retVal = "";
            retVal += "\nmax=" + NeuralNet.s(max);
            retVal += "\nmin=" + NeuralNet.s(min);

            retVal += "\navg=" + NeuralNet.s(avg);
            retVal += "\naavg=" + NeuralNet.s(aavg);
            retVal += "\nn=" + n;

            return retVal;
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

    /**
     * return a map of value->index for all values found in given column
     */
    public static Map<String, Integer> columnValueIndex(List<String[]> dataCsv, int categoryIdx) {
        Map<String, Integer> categories = new HashMap<String, Integer>();
        for (String[] row : dataCsv) {
            categories.putIfAbsent(row[categoryIdx], categories.size());
        }
        return categories;
    }

    /**
     * create a category array, i.e. zero filled with only k set to 1
     */
    public static double[] toArgmax(int size, int idx) {
        double[] retVal = new double[size];
        retVal[idx] = 1;
        return retVal;
    }


    /**
     * this is the opposite of toArgmax, it returns the index of maximum value found in the array
     */
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

    /**
     * adjust values in column c of input, so that all values are in [0;1]
     */
    static void normalizeColumn(List<InAndOut> data, int c) {
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
