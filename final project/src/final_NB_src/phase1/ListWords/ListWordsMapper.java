package ListWords;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class ListWordsMapper extends Mapper<Object, Text, Text, IntWritable> {
    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        //map读取TF步骤的所有文件，遇到一个词就发射<词，次数1>。不需要Combine操作，因为TF生成的文件中每个词必定只出现了一次
        Text word = new Text(value.toString().split("\t")[0]);
        context.write(new Text(word), new IntWritable(1));
    }
}
