package Count;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class CountCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
    //输入是<文件名#单词，1>
    //输出是<文件名#单词，词频>

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        //int splitSymbolIndex = key.toString().indexOf("#");
        //计算词频除以总词数，即TF值
        int tf = 0;
        for (IntWritable val : values) {
            tf += val.get();
        }
        context.write(key, new IntWritable(tf));
    }
}
