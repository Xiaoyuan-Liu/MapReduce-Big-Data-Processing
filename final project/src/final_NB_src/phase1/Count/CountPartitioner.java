package Count;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

public class CountPartitioner extends HashPartitioner<Text, IntWritable> {
    //输入是<文件类/文件名#单词，词频>
    //输出是以<文件名,词频>划分的Reduce服务器号

    @Override
    public int getPartition(Text key, IntWritable value, int numReduceTasks) {
        String term = key.toString().split("#")[0];
        //System.err.println(term);
        return super.getPartition(new Text(term), value, numReduceTasks);
    }
}
