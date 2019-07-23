package Accuracy;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class AccuracyMapper extends Mapper<Object, Text, Text, DoubleWritable> {

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
        String[] line = value.toString().split("\t");
        String name = line[0];
        String predict = line[1];
        String cls = name.split("#")[0];

        if(cls.equals(predict)){
            context.write(new Text("T"), new DoubleWritable(1));
        }
        else{
            context.write(new Text("F"), new DoubleWritable(1));
        }
    }

}
