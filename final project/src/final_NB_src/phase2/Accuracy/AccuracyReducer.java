package Accuracy;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class AccuracyReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private double allT = 0;
    private double allF = 0;

    @Override
    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException{
        if(key.toString().equals("T")){
            for(DoubleWritable val: values){
                allT += val.get();
            }
        }
        else{
            for(DoubleWritable val :values){
                allF += val.get();
            }
        }
    }

    public void cleanup(Context context)
            throws IOException, InterruptedException{
        context.write(new Text(" "), new DoubleWritable(allT/(allF + allT)));
    }
}
