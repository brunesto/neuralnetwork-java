package bruno.nn;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.ListSelectionModel;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.ToolTipManager;
import javax.swing.border.Border;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableModel;

import bruno.nn.Helper.InAndOut;
import bruno.nn.Helper.TrainConfig;
import bruno.nn.SwingViewer.ArrayPanel.ModelConfig;
import bruno.nn.SwingViewer.ArrayPanel.Stats;

public class SwingViewer {
    NeuralNet nn;

    public SwingViewer(NeuralNet nn) {
        this.nn = nn;
    }

    /**
     * copied from Oracle samples
     */
    public static class ColorRenderer extends JLabel
            implements TableCellRenderer {
        Border unselectedBorder = null;
        Border selectedBorder = null;
        boolean isBordered = true;
        String prefix;

        Font font;

        Supplier<ModelConfig> getModelConfig;

        public ColorRenderer(boolean isBordered, String prefix, Supplier<ModelConfig> getModelConfig) {
            this.isBordered = isBordered;
            this.prefix = prefix;
            this.getModelConfig = getModelConfig;

            setOpaque(true);
            font = new Font(font.MONOSPACED, Font.PLAIN, 8);

        }

        public Component getTableCellRendererComponent(
                JTable table, Object vo,
                boolean isSelected, boolean hasFocus,
                int row, int column) {

            ModelConfig modelConfig = getModelConfig.get();

            int i = (row * modelConfig.cols + column) + modelConfig.startIdx;
            if (i >= 0 && i < modelConfig.endIdx) {
                setFont(font);
                double v = modelConfig.vs[i];
                Color newColor = ArrayPanel.v2color(v);

                setForeground(v > -0.4 ? Color.BLACK : Color.WHITE);
                setBackground(newColor);
                if (isBordered) {
                    if (isSelected) {
                        if (selectedBorder == null) {
                            selectedBorder = BorderFactory.createMatteBorder(1, 1, 1, 1,
                                    table.getSelectionBackground());
                        }
                        setBorder(selectedBorder);
                    } else {
                        if (unselectedBorder == null) {
                            unselectedBorder = BorderFactory.createMatteBorder(1, 1, 1, 1,
                                    table.getBackground());
                        }
                        setBorder(unselectedBorder);
                    }
                }
                setText(String.format(NeuralNet.s(v)));
                setToolTipText(prefix + "[" + i + "]" + "=" + String.format("%5.10f", v));

            } else {
                setBackground(Color.white);
                setToolTipText("");
                setText("");

            }

            return this;
        }
    }

    static class ArrayPanel extends JPanel {

        int selectedIdx = -1;
        int hoverIdx = -1;
        int size = 20;

        int startx = 10;
        int starty = 30;

        Consumer<Integer> onHover;
        Consumer<Integer> onSelected;

        //        public int realIdx2pixx(int i) {
        //            return (i - startIdx) % cols + startx;
        //        }

        JTextArea stats;
        JTable table;
        TableModel tableModel;

        ModelConfig modelConfig;

        static class ModelConfig {
            double vs[];
            int cols;
            int startIdx;
            int endIdx;

            public ModelConfig(double[] vs, int cols, int startIdx, int endIdx) {
                super();
                this.vs = vs;
                this.cols = cols;
                this.startIdx = startIdx;
                this.endIdx = endIdx;
            }

            int maxLenght;
            int maxy;

        }

        public void setNewModel(ModelConfig modelConfig) {
            if (this.modelConfig == null)
                this.modelConfig = modelConfig;
            if (modelConfig.vs == null)
                this.modelConfig.vs = new double[0];
            else
                this.modelConfig.vs = modelConfig.vs;
            //                throw new NullPointerException();
            if (modelConfig.cols != -1)
                modelConfig.cols = modelConfig.cols;
            this.modelConfig.startIdx = modelConfig.startIdx;
            if (modelConfig.endIdx == -1)
                modelConfig.endIdx = this.modelConfig.vs.length;
            this.modelConfig.endIdx = modelConfig.endIdx;
            this.modelConfig.maxLenght = this.modelConfig.endIdx - this.modelConfig.startIdx;
            //System.err.println("" + maxLenght);
            this.modelConfig.maxy = 1 + (this.modelConfig.maxLenght / this.modelConfig.cols);

            //            Dimension d = new Dimension(2 * startx + this.cols * size, 2 * starty + maxy * size);
            //            setPreferredSize(d);
            //            setMinimumSize(d);
            onModelChanged();

        }

        private void onModelChanged() {
            //            System.err.println("onModelChanged " + title);
            if (this.modelConfig != null && this.modelConfig.vs != null && this.modelConfig.vs.length > 0) {
                Stats stats2 = new Stats(this.modelConfig.vs);
                stats.setText("" + stats2);

            }
            // invalidate();
            repaint();

        }

        //        public int idx2pixy(int i) {
        //            return (i - startIdx) / cols + starty;
        //        }

        //        public int pix2realIdx(int pixx, int pixy) {
        //
        //            int x = (pixx - startx) / size;
        //            int y = (pixy - starty) / size;
        //
        //            if (x >= 0 && x < cols)
        //                if (y >= 0 && y <= maxLenght / cols) {
        //                    int idx = y * cols + x + startIdx;
        //                    if (idx < maxLenght)
        //                        return idx;
        //                }
        //            return -1;
        //
        //        }

        public void setOnHover(Consumer<Integer> onHover) {
            this.onHover = onHover;
        }

        public void setOnSelected(Consumer<Integer> onSelected) {
            this.onSelected = onSelected;
        }

        String title;
        ColorRenderer colorRenderer;

        public void setTitle(String title) {
            this.title = title;
            colorRenderer.prefix = title;
        }

        public ArrayPanel(String title, double vs[], int cols) {
            super(new BorderLayout());
            ToolTipManager.sharedInstance().setInitialDelay(100);
            ToolTipManager.sharedInstance().setReshowDelay(200);
            this.title = title;
            stats = new JTextArea();
            add(stats, BorderLayout.NORTH);
            stats.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 8));
            table = new JTable() {
                @Override
                public Dimension getPreferredSize() {
                    return super.getMinimumSize();
                }

            };

            add(table, BorderLayout.CENTER);

            tableModel = new AbstractTableModel() {

                @Override
                public int getRowCount() {
                    //                    System.err.println(title + " modelConfig.maxy:" + modelConfig.maxy);
                    return modelConfig.maxy;
                }

                @Override
                public int getColumnCount() {
                    return cols;
                }

                @Override
                public Object getValueAt(int rowIndex, int columnIndex) {
                    int idx = rowIndex * cols + columnIndex + modelConfig.startIdx;
                    if (idx < modelConfig.maxLenght)
                        return modelConfig.vs[idx];
                    return null;
                }

            };
            table.addMouseListener(new MouseAdapter() {

                @Override
                public void mouseClicked(MouseEvent e) {
                    int y = table.getSelectedColumn();
                    int x = table.getSelectedRow();
                    System.err.println("selected " + x + "," + y);
                    int idx = y * cols + x;
                    if (idx < modelConfig.maxLenght) {
                        if (onSelected != null)
                            onSelected.accept(idx + modelConfig.startIdx);
                    }
                }
            });

            table.setModel(tableModel);
            table.setCellSelectionEnabled(true);
            table.getSelectionModel().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

            colorRenderer = new ColorRenderer(true, title, () -> modelConfig);
            table.setDefaultRenderer(Object.class, colorRenderer);
            for (int i = 0; i < cols; i++)
                table.getColumnModel().getColumn(i).setMaxWidth(40);
            table.getColumnModel().setColumnMargin(1);

            setNewModel(new ModelConfig(vs, cols, 0, -1));

            setBorder(BorderFactory.createTitledBorder(title));
            //            this.addMouseMotionListener(new MouseMotionAdapter() {
            //
            //                public void mouseMoved(MouseEvent e) {
            //
            //                    int idx = pix2realIdx(e.getX(), e.getY());
            //
            //                    if (onHover != null)
            //                        onHover.accept(idx);
            //
            //                    hoverIdx = idx;
            //                    // System.err.println("hoverI:" + hoverIdx);
            //                    repaint();
            //
            //                }
            //
            //            });

            //            this.addMouseListener(new MouseAdapter() {
            //
            //                @Override
            //                public void mouseClicked(MouseEvent e) {
            //                    if (onSelected != null) {
            //                        int idx = pix2realIdx(e.getX(), e.getY());
            //                        if (idx != -1) {
            //                            setSelected(idx);
            //                            onSelected.accept(idx);
            //                        }
            //                    }
            //
            //                }
            //            });
        }

        public void setSelected(int i) {
            selectedIdx = i;
            System.err.println("selectedI:" + selectedIdx);

            repaint();
        }

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

        //        public void paintComponent(Graphics g) {
        //            try {
        //                super.paintComponent(g);
        //                // Draws the image to the canvas
        //                System.err.println("paintComponent() selectedI:" + selectedIdx);
        //
        //                g.clearRect(0, 0, getWidth(), getHeight());
        //
        //                for (int i = 0; i < maxLenght; i++) {
        //                    int x = i % cols;
        //                    int y = i / cols;
        //                    int idx = i + startIdx;
        //                    if (idx == hoverIdx) {
        //                        g.setColor(Color.YELLOW);
        //                        g.fillRect(startx + x * size - 2, starty + y * size - 2, size + 2, size + 2);
        //                    }
        //                    if (idx == selectedIdx) {
        //                        g.setColor(Color.BLUE);
        //                        g.fillRect(startx + x * size - 2, starty + y * size - 2, size + 2, size + 2);
        //                    }
        //
        //                }
        //
        //                for (int i = 0; i < maxLenght; i++) {
        //
        //                    int x = i % cols;
        //                    int y = i / cols;
        //                    int idx = i + startIdx;
        //                    double v = vs[idx];
        //                    g.setColor(v2color(v));
        //                    g.fillRect(startx + x * size, starty + y * size, size - 2, size - 2);
        //                }
        //            } catch (Exception e) {
        //                throw new RuntimeException(e);
        //            }
        //        }

        public static Color v2color(double v) {
            // v = Math.sqrt(v * 10);
            int r = 0, g = 0, b = 0;

            if (v == 0) {
                r = 0xff;
                g = 0xff;
                b = 0xff;
            } else {
                //[-1;+1] ->[0;1]
                v = (v + 1) / 2;
                if (v > 1)
                    v = 1;
                if (v < 0)
                    v = 0;

                r = Math.min(255, (int) (v * 256));
                b = Math.min(255, (int) ((1 - v) * 256));
                double v2 = Math.abs(v - 0.5) * 2;
                g = Math.min(255, (int) ((1 - v2) * 128));
            }
            int rgb = (r << 16) + (g << 8) + b;
            return new Color(rgb);
        }

    }

    static class ArrayPanels extends JPanel {
        List<ArrayPanel> arrayPanels = new ArrayList<>();

        public ArrayPanels() {
            this.setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        }

        public void onModelChanged() {
            for (ArrayPanel arrayPanel : arrayPanels)
                arrayPanel.onModelChanged();
        }

        public void addArrayPanel(ArrayPanel arrayPanel) {
            super.add(arrayPanel);
            arrayPanels.add(arrayPanel);
        }
    }

    static class LayerPanel extends JPanel {
        NeuralNet nn;

        int l;
        // selected neuron
        int i;
        //        JLabel label;
        List<Integer> dims;

        public JComponent addBackward() {

            bwd = new ArrayPanels();
            bwd.setBorder(BorderFactory.createTitledBorder("bwd"));

            prevd = new ArrayPanel("dls[" + (l - 1) + "]", nn.dls[l - 1], dims.get(l - 1));
            //            prevd.setOnHover(x -> {
            //                if (x != -1)
            //                    label.setText("dls[" + (l - 1) + "][" + x + "]:" + nn.dls[l - 1][x]);
            //                else
            //                    label.setText("");
            //            });

            bwd.addArrayPanel(prevd);//, BorderLayout.WEST);

            //            JPanel center = new JPanel(new BorderLayout());
            //            panel.add(center, BorderLayout.CENTER);

            weightsd = new ArrayPanel("dws[" + (l - 1) + "][0]", nn.dws[l - 1][0], dims.get(l - 1));
            //            weightsd.setOnHover(x -> {
            //                if (x != -1 && i != -1)
            //                    label.setText("dws[" + (l - 1) + "][" + i + "][" + x + "]:" + nn.dws[l - 1][i][x]);
            //                else
            //                    label.setText("");
            //            });

            bwd.addArrayPanel(weightsd);//, BorderLayout.CENTER);

            biasd = new ArrayPanel("dbs[" + (l - 1) + "][0]", nn.dbs[l - 1], 1);
            biasd.setNewModel(new ModelConfig(nn.dbs[l - 1], 1, 0, 1));
            //            biasd.setOnHover(x -> {
            //                // x is always 0
            //                if (x != -1 && i != -1)
            //                    label.setText("dbs[" + (l - 1) + "][" + i + "]:" + nn.dbs[l - 1][i]);
            //                else
            //                    label.setText("");
            //            });

            bwd.addArrayPanel(biasd);//, BorderLayout.EAST);

            currentd = new ArrayPanel("dls[" + (l) + "]", nn.dls[l], dims.get(l));
            currentd.setOnSelected(this::selectNeuron);

            bwd.add(currentd, BorderLayout.EAST);

            JScrollPane scrollPane = new JScrollPane(bwd);
            return scrollPane;
        }

        ArrayPanel prev;
        ArrayPanel weights;
        ArrayPanel bias;
        ArrayPanel zs;
        ArrayPanel current;

        ArrayPanel prevd;
        ArrayPanel weightsd;
        ArrayPanel biasd;

        ArrayPanel currentd;
        ArrayPanels fwd;
        ArrayPanels bwd;

        public JComponent addForward() {
            fwd = new ArrayPanels();
            fwd.setBorder(BorderFactory.createTitledBorder("fwd"));
            prev = new ArrayPanel("a[" + (l - 1) + "]", nn.ls[l - 1], dims.get(l - 1));
            //            prev.setOnHover(x -> {
            //                if (x != -1)
            //                    label.setText("a[" + (l - 1) + "][" + x + "]:" + nn.ls[l - 1][x]);
            //                else
            //                    label.setText("");
            //            });

            fwd.addArrayPanel(prev);//, BorderLayout.WEST);

            //            JPanel center = new JPanel(new BorderLayout());
            //            panel.add(center, BorderLayout.CENTER);

            weights = new ArrayPanel("ws[" + (l - 1) + "][0]", nn.ws[l - 1][0], dims.get(l - 1));
            //            weights.setOnHover(x -> {
            //                if (x != -1 && i != -1)
            //                    label.setText("ws[" + (l - 1) + "][" + i + "][" + x + "]:" + nn.ws[l - 1][i][x]);
            //                else
            //                    label.setText("");
            //            });

            fwd.addArrayPanel(weights);//, BorderLayout.CENTER);

            bias = new ArrayPanel("bs[" + (l - 1) + "][0]", nn.bs[l - 1], 1);
            bias.setNewModel(new ModelConfig(nn.bs[l - 1], 2, 0, 1));
            //            bias.setOnHover(x -> {
            //                // x is always 0
            //                if (x != -1 && i != -1)
            //                    label.setText("bs[" + (l - 1) + "][" + i + "]:" + nn.bs[l - 1][i]);
            //                else
            //                    label.setText("");
            //            });

            fwd.addArrayPanel(bias);//, BorderLayout.EAST);

            zs = new ArrayPanel("zs[" + (l) + "]", nn.zs[l], dims.get(l));
            zs.setOnSelected(this::selectNeuron);

            fwd.addArrayPanel(zs);

            current = new ArrayPanel("a[" + (l) + "]", nn.ls[l], dims.get(l));
            current.setOnSelected(this::selectNeuron);

            fwd.addArrayPanel(current);
            JScrollPane scrollPane = new JScrollPane(fwd);
            return scrollPane;
        }

        private void selectNeuron(int i) {

            weights.setNewModel(new ModelConfig(nn.ws[l - 1][i], -1, 0, -1));
            weights.setTitle("ws[" + (l - 1) + "][" + i + "]");
            bias.setNewModel(new ModelConfig(nn.bs[l - 1], -1, i, i + 1));
            bias.setTitle("bs[" + (l - 1) + "][" + i + "]");
            bias.setBorder(BorderFactory.createTitledBorder("bs[" + (l - 1) + "][" + i + "]"));
            weights.setBorder(BorderFactory.createTitledBorder("ws[" + (l - 1) + "][" + i + "]"));

            weightsd.setNewModel(new ModelConfig(nn.dws[l - 1][i], -1, 0, -1));
            weightsd.setTitle("dws[" + (l - 1) + "][" + i + "]");
            biasd.setNewModel(new ModelConfig(nn.dbs[l - 1], -1, i, i + 1));
            biasd.setTitle("dbs[" + (l - 1) + "][" + i + "]");
            biasd.setBorder(BorderFactory.createTitledBorder("dbs[" + (l - 1) + "][" + i + "]"));
            weightsd.setBorder(BorderFactory.createTitledBorder("dws[" + (l - 1) + "][" + i + "]"));
        }

        public LayerPanel(NeuralNet nn, int l, List<Integer> dims) {
            super(new BorderLayout());
            this.nn = nn;
            this.l = l;
            this.dims = dims;

            //            label = new JLabel("click on left side neurons", JLabel.CENTER);
            //            this.add(label, BorderLayout.SOUTH);

            JSplitPane jSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
            jSplitPane.setResizeWeight(0.5);
            this.add(jSplitPane, BorderLayout.CENTER);

            //            JPanel fwdAndCost = new JPanel(new BorderLayout());
            //            fwdAndCost.add(addForward(), BorderLayout.CENTER);

            //            JLabel costLabel = new JLabel();
            //
            //            fwdAndCost.add(costLabel);

            jSplitPane.add(addForward());
            jSplitPane.add(addBackward());

        }

        public void onModelChanged() {
            fwd.onModelChanged();
            bwd.onModelChanged();

        }

    }

    static class CostPanel extends JPanel {
        private static final long serialVersionUID = 1L;

        double errorRef[];
        double errordRef[];
        NeuralNet nn;
        ArrayPanels bwd;
        ArrayPanels fwd;
        JLabel costLabel;

        public CostPanel(NeuralNet nn, int dim) {
            super(new BorderLayout());
            this.setBorder(BorderFactory.createTitledBorder("cost"));
            this.nn = nn;
            costLabel = new JLabel();
            this.add(costLabel, BorderLayout.NORTH);


            JSplitPane jSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
            this.add(jSplitPane, BorderLayout.CENTER);

            fwd = new ArrayPanels();
            fwd.setBorder(BorderFactory.createTitledBorder("fwd"));
            jSplitPane.add(new JScrollPane(fwd));

            ArrayPanel output = new ArrayPanel("a[" + (nn.layers - 1) + "]", nn.ls[(nn.layers - 1)], dim);
            fwd.addArrayPanel(output);

            errorRef = new double[nn.config.layer_sizes[nn.config.layer_sizes.length - 1]];
            ArrayPanel error = new ArrayPanel("error", errorRef, dim);
            fwd.addArrayPanel(error);

            bwd = new ArrayPanels();
            bwd.setBorder(BorderFactory.createTitledBorder("bwd"));
            jSplitPane.add(new JScrollPane(bwd));
            errordRef = new double[nn.config.layer_sizes[nn.config.layer_sizes.length - 1]];
            ArrayPanel errord = new ArrayPanel("errord", errordRef, dim);
            bwd.addArrayPanel(errord);

        }

        public void reset() {
            NeuralNet.zeros(errorRef);
            NeuralNet.zeros(errordRef);
        }

        public void computeError(double expected[]) {
            Helper.layerErrorFunction(nn.ls[(nn.layers - 1)], expected, errorRef);

            costLabel.setText("$ " + NeuralNet.s(new Stats(errorRef).acc));

            Helper.layerErrorFunctiond(nn.ls[(nn.layers - 1)], expected, errordRef);
            invalidate();
            repaint();
        }

        public void onModelChanged(double expected[]) {
            computeError(expected);
            fwd.onModelChanged();
            bwd.onModelChanged();

        }

    }

    public static List<Integer> getDims(int n, int... dims) {
        List<Integer> retVal = new ArrayList<Integer>();
        for (int i = 0; i < dims.length; i++)
            retVal.add(dims[i]);
        while (retVal.size() < n)
            retVal.add(1);

        return retVal;
    }

    static class NnPanel extends JPanel {
        private static final long serialVersionUID = 1L;

        //        JLabel costLabel;
        List<LayerPanel> layerPanels = new ArrayList<>();

        public NnPanel(NeuralNet nn, List<Integer> dims) {
            super(new BorderLayout());

            JTabbedPane jTabbedPane = new JTabbedPane();
            this.add(jTabbedPane, BorderLayout.CENTER);

            for (int l = 1; l < nn.config.layer_sizes.length; l++) {
                LayerPanel layerPanel = new LayerPanel(nn, l, dims);
                layerPanels.add(layerPanel);
                jTabbedPane.add("" + l, layerPanel);
            }
            //            costLabel = new JLabel();
            //            jTabbedPane.add("cost", costLabel);
        }

        public void onModelChanged() {
            for (LayerPanel layerPanel : layerPanels)
                layerPanel.onModelChanged();

        }
    }

    static class InAndOutPanel extends JPanel {
        private static final long serialVersionUID = 1L;
        List<InAndOut> list;
        JSpinner spinner;
        ArrayPanel in;
        ArrayPanel out;
        BiConsumer<Integer, Boolean> onSelected;
        JCheckBox autoEvaluate;

        public InAndOutPanel(List<InAndOut> list, int dimIn, int dimOut) {
            super(new BorderLayout());
            this.list = list;

            JPanel chooseSamplePanel = new JPanel();
            this.add(chooseSamplePanel, BorderLayout.NORTH);

            chooseSamplePanel.add(new JLabel("sample (max:" + (list.size() - 1) + "):"));
            SpinnerModel model = new SpinnerNumberModel(0, //initial value
                    0,
                    list.size() - 1,
                    1); //step
            spinner = new JSpinner(model);
            chooseSamplePanel.add(spinner);

            JSplitPane jTabbedPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
            this.add(jTabbedPane, BorderLayout.CENTER);

            in = new ArrayPanel("X", list.get(0).input, dimIn);
            jTabbedPane.add(new JScrollPane(in));
            out = new ArrayPanel("Y", list.get(0).expected, dimOut);
            jTabbedPane.add(new JScrollPane(out));

            spinner.addChangeListener(x -> {
                int i = (int) spinner.getValue();
                in.setNewModel(new ModelConfig(list.get(i).input, dimIn, 0, -1));
                out.setNewModel(new ModelConfig(list.get(i).expected, dimOut, 0, -1));
                if (onSelected != null)
                    onSelected.accept(i, autoEvaluate.isSelected());

            });

            autoEvaluate = new JCheckBox("auto fwd-bwd");
            chooseSamplePanel.add(autoEvaluate);
        }

        int getSelected() {
            return (int) spinner.getValue();
        }

        public void setOnSelected(BiConsumer<Integer, Boolean> onSelected) {
            this.onSelected = onSelected;
        }
    }

    static class MainPanel extends JPanel {

        private static final long serialVersionUID = 1L;

        JLabel statusBar;
        NeuralNet nn;
        JButton applyBtn;
        NnPanel nnPanel;
        InAndOutPanel inOutPanel;
        CostPanel costPanel;
        List<InAndOut> data;

        public MainPanel(NeuralNet nn, List<InAndOut> data, List<Integer> dims) {
            super(new BorderLayout());
            this.nn = nn;
            this.data = data;

            setPreferredSize(new Dimension(500, 500));
            setBackground(Color.WHITE);

            JPanel toolBar = new JPanel();
            this.add(toolBar, BorderLayout.NORTH);

            statusBar = new JLabel("...");
            this.add(statusBar, BorderLayout.SOUTH);
            JButton resetBtn = new JButton("reset");
            toolBar.add(resetBtn);

            JButton dw0Btn = new JButton("dw0");
            toolBar.add(dw0Btn);
            JButton backtrackBtn = new JButton("backtrack");
            toolBar.add(backtrackBtn);

            applyBtn = new JButton();
            toolBar.add(applyBtn);

            JButton train1Btn = new JButton("U+A");
            toolBar.add(train1Btn);

            JButton normBtn = new JButton("norm");
            toolBar.add(normBtn);

            JButton batchBtn = new JButton("batch");
            toolBar.add(batchBtn);

            JSplitPane jSplitPane1 = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
            jSplitPane1.setResizeWeight(0.25);
            this.add(jSplitPane1, BorderLayout.CENTER);

            inOutPanel = new InAndOutPanel(data, dims.get(0), dims.get(dims.size() - 1));
            jSplitPane1.add(inOutPanel);
            inOutPanel.setOnSelected((Integer i, Boolean b) -> {
                if (b)
                    backtrack(nn, data);
            });

            nn.resetDws();

            JSplitPane jSplitPane2 = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
            jSplitPane2.setResizeWeight(0.75);
            jSplitPane1.add(jSplitPane2);
            nnPanel = new NnPanel(nn, dims);
            jSplitPane2.add(nnPanel);

            costPanel = new CostPanel(nn, dims.get(dims.size() - 1));
            jSplitPane2.add(costPanel);

            resetBtn.addActionListener(e -> {
                nn.reset();
                onModelChanged();
                costPanel.reset(); // not necessary, just for the UI

            });

            dw0Btn.addActionListener(e -> {
                resetDw(nn, costPanel);

            });
            backtrackBtn.addActionListener(e -> {
                backtrack(nn, data);
            });
            applyBtn.addActionListener(e -> {
                nn.applyDws();
                resetDw(nn, costPanel);

            });
            train1Btn.addActionListener(e -> {
                resetDw(nn, costPanel);
                backtrack(nn, data);
                nn.applyDws();
                resetDw(nn, costPanel);
            });
            normBtn.addActionListener(e -> {
                nn.normalizeWs();
                onModelChanged();
            });

            batchBtn.addActionListener(e -> {
                TrainConfig trainConfig = new TrainConfig();
                trainConfig.epochs = 1;
                trainConfig.rateDecay = 1.0;
                trainConfig.batches = 1;
                //trainConfig.reduceTrainingRatio = 0.1;
                Helper.train(data, data, nn, trainConfig);
                onModelChanged();
                double[] metrics=Helper.computeErrorAcc(data,nn);
                statusBar.setText("after batch. avg $" + NeuralNet.s(metrics[0]) + " avg accuracy:" + NeuralNet.s(metrics[1]));
                
            });

            updateApplyButtonTitle();
        }

        private void backtrack(NeuralNet nn, List<InAndOut> data) {
            int i = inOutPanel.getSelected();
            InAndOut sample = data.get(i);
            double[] output = nn.computeNetwork(sample.input);
            nn.updateBacktrack(sample.input, sample.expected);
            double[] pair = Helper.computeError(output, sample.expected);
            statusBar.setText("sample #" + i + " cost:" + pair[0] + " accuracy" + pair[1]);

            costPanel.computeError(sample.expected);

            updateApplyButtonTitle();
            onModelChanged();
        }

        private void resetDw(NeuralNet nn, CostPanel costPanel) {

            //            nn.resetDls(); // not necessary, just for the UI
            nn.resetDws();

            updateApplyButtonTitle();
            onModelChanged();
        }

        private void onModelChanged() {
            nnPanel.onModelChanged();
            costPanel.onModelChanged(data.get(inOutPanel.getSelected()).expected);
            repaint();
        }

        private void updateApplyButtonTitle() {
            applyBtn.setText("apply (h=" + nn.dh + ")");
            applyBtn.setEnabled(nn.dh > 0);
        }

    }

    public static void show(NeuralNet nn, List<InAndOut> list, int... dims) {
        EventQueue.invokeLater(() -> {
            JFrame f = new JFrame();
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.add(new MainPanel(nn, list, getDims(nn.layers, dims)));
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
    }

}
