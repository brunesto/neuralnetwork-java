package bruno.nn;

import java.io.IOException;
import java.math.RoundingMode;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Locale;

/**
 * simple csv parser/writer. It is meant to be compatible with CsvParser.py
 */
public class CsvHelper {

    // -- ensures java output matches python's ----------------------

    static NumberFormat nfd;
    static {
        nfd = NumberFormat.getNumberInstance(Locale.US);
        nfd.setMaximumFractionDigits(20);
        nfd.setRoundingMode(RoundingMode.HALF_EVEN);
    }

    static NumberFormat nfs = new DecimalFormat("0.##################E00");

    static String formatAlaPython(double v) {
        if (Math.abs(v) < 1e-04)
            return nfs.format(v).replace('E', 'e');
        else
            return nfd.format(v);
    }

    public static void main(String... args) {

        System.err.println(formatAlaPython(7.563140912147937e-05));
        System.err.println(formatAlaPython(0.022601457442347928));
        System.err.println(formatAlaPython(-0.00031424492913956215));
    }

    //-- consumming i.e. parsing      

    public static int consumeComments(String lines[], int y) {
        while (lines[y].startsWith("#") || lines[y].isBlank()) {
            y++;
        }

        return y;
    }

    public static int consume1d(String lines[], int y, double[] ds) {
        y = consumeComments(lines, y);
        String tokens[] = lines[y].split(",");
        y++;
        for (int i = 0; i < ds.length; i++) {
            ds[i] = Double.parseDouble(tokens[i]);
        }
        return y;

    }

    public static int consume2d(String[] content, int y, double[][] ds) {
        for (int i = 0; i < ds.length; i++)
            y = consume1d(content, y, ds[i]);
        return y;
    }

    public static void consumeFile(NeuralNet nn, String path) throws IOException {
        String lines[] = Files.readString(Paths.get(path)).split("\n");
        consumeNeuralNet(nn, lines);
    }

    public static void consumeNeuralNet(NeuralNet nn, String lines[]) throws IOException {
        int y = 0;
        for (int l = 0; l < nn.config.layer_sizes.length - 1; l++) {
            y = consume2d(lines, y, nn.ws[l]);
            y = consume1d(lines, y, nn.bs[l]);
        }
    }

    // -- dumping ,i.e. formatting ------------------------------------------------
    public static String dump1d(double[] vs, String info) {
        String s = info + (info.isBlank() ? "" : "\n");
        for (int i = 0; i < vs.length; i++) {
            if (i > 0)
                s = s + ", ";
            s = s + formatAlaPython(vs[i]);

        }
        return s;
    }

    public static String dump2d(double[][] vss, String info) {
        String s = info + (info.isBlank() ? "" : "\n");
        for (int i = 0; i < vss.length; i++) {
            if (i > 0)
                s = s + "\n";
            s = s + dump1d(vss[i], info.isBlank() ? "" : (info + "[" + (i) + "]"));
        }
        return s;
    }

    public static String dump3d(double[][][] vsss, String info) {
        String s = info + (info.isBlank() ? "" : "\n");
        for (int i = 0; i < vsss.length; i++) {
            if (i > 0)
                s = s + "\n";
            s = s + dump2d(vsss[i], info.isBlank() ? "" : (info + "[" + (i) + "]"));
        }
        return s;
    }

    public static String dumpNeuralNet(NeuralNet nn) {
        String csv = "";
        for (int l = 0; l < nn.config.layer_sizes.length - 1; l++) {
            csv = csv + "\n# layer " + (l);
            csv = csv + "\n" + dump2d(nn.ws[l], "#ws[" + (l) + "]");
            csv = csv + "\n" + dump1d(nn.bs[l], "#bs[" + (l) + "]");
        }
        return csv;
    }

    public static void dumpNeuralNetToFile(NeuralNet nn, String path) throws IOException {
        Files.write(Paths.get(path), dumpNeuralNet(nn).getBytes());
    }

}