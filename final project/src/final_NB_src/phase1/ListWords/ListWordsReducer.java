package ListWords;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class ListWordsReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        //每个词出现的总词数除以总文件数就是IDF值
        int wordCnt = 0;
        for(IntWritable val:values){
            wordCnt += val.get();
        }
        context.write(new Text(key.toString()), new IntWritable(wordCnt));
    }
}
