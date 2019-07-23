import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class AccuracyCombiner extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    @Override
    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException{
        double cnt = 0;
        for(DoubleWritable val : values){
            cnt += val.get();
        }
        context.write(key, new DoubleWritable(cnt));
    }
}