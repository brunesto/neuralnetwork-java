package bruno.nn;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

/**
 * Class to read the Mnist samples
 * Warning: values differ slightly from the python equivalent
 */
public class ReadMnist {
    private String imagesFile;
    private String labelsFile;
    public int[] labels;
    public List<double[]> images;

    public ReadMnist(String imagesFile, String labelsFile) {
        this.imagesFile = imagesFile;
        this.labelsFile = labelsFile;
        readLabels();
        readImages();
    }


    private long extract(byte[] container, int i, int s) {
        long acc = 0;
        for (int d = 0; d < s; d++) {
            acc *= 256;
            acc += (container[i + d] & 0xFF);
        }
        return acc;
    }

    private void readLabels() {
        byte[] t10kLabels = readFile(labelsFile);
        int magic = (int) extract(t10kLabels, 0, 4);
        if (magic != 0x00000801) {
            throw new IllegalArgumentException("" + magic);
        }
        int size = (int) extract(t10kLabels, 4, 4);
        System.out.println("size:" + size);
        labels = new int[size];
        for (int i = 0; i < size; i++) {
            labels[i] = (t10kLabels[i + 8] & 0xFF);
        }
    }

    private void readImages() {
        byte[] t10kImages = readFile(imagesFile);
        int magic = (int) extract(t10kImages, 0, 4);
        if (magic != 0x00000803) {
            throw new IllegalArgumentException("" + magic);
        }
        int size = (int) extract(t10kImages, 4, 4);
        int x = (int) extract(t10kImages, 8, 4);
        int y = (int) extract(t10kImages, 12, 4);
        System.out.println("size:" + size + ",x:" + x + " y:" + y);
        images = new ArrayList<>();
        int c = 16;
        for (int i = 0; i < size; i++) {
            double[] image = new double[x * y];
            for (int p = 0; p < x * y; p++) {
                int ub = (t10kImages[c] & 0xFF); // unsigned byte 0...255
                image[p] = ub / 255.0; // [0..1]
                c++;
            }
            images.add(image);
        }
    }

    private byte[] readFile(String fileName) {
        try {
            return Files.readAllBytes(Paths.get(fileName));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void saveAsPng(String fileNameRoot, int n) {
        int w = 28;
        int h = 28;
        BufferedImage buffedImage = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        double[] image = images.get(n);
        int label = labels[n];
        int c = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = (int) ((1 - image[c]) * 65535);
                buffedImage.setRGB(x, y, v);
                c++;
            }
        }
        try {
            ImageIO.write(buffedImage, "png", new File(fileNameRoot + "" + n + "-" + label + ".png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        ReadMnist r = new ReadMnist("/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-images-idx3-ubyte", "/home/bc2/bruno/work/github/brunesto/neuralnetwork-py/data/t10k-labels-idx1-ubyte");
        for (int i = 0; i <= 20; i++) {
            r.saveAsPng("/tmp/", i);
        }
    }
}