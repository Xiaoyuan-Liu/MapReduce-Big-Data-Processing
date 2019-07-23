package Count;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class CountReducer extends Reducer<Text, IntWritable, NullWritable, Text> {
    private Text word1 = new Text();
    private Text word2 = new Text();
    String temp = new String();
    static Text CurrentItem = new Text(" ");
    static List<String> postingList = new ArrayList<String>();

    //使用MUltipleOutputs类，输出为多文件
    private MultipleOutputs<NullWritable, Text> multipleOutputs = null;

    @Override
    public void setup(Context context)
            throws IOException, InterruptedException {
        multipleOutputs = new MultipleOutputs<NullWritable, Text>(context);
    }

    //reduce输出的key-value对格式:term <doc1,num1>...<total,sum>
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        String fileName = key.toString().split("#")[0];
        String filePath = fileName.split("/")[0];
        fileName = fileName.split("/")[1];
        String word = key.toString().split("#")[1];
        //System.err.println(key.toString());
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(10);
        for (IntWritable val : values)
            //输出文件名为“类别#原文件名”，每行为一个单词及其TF值，用制表符隔开
            multipleOutputs.write(NullWritable.get(), new Text(word + "\t" + val.get()), filePath+ "/" + context.getProfileParams() + "#" + fileName);
    }

    public void cleanup(Reducer.Context context)
            throws IOException, InterruptedException {
        if (null != multipleOutputs) {
            multipleOutputs.close();
            multipleOutputs = null;
        }
    }
}

