package TrainModel;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class TrainModelReducer extends Reducer<Text, Text, Text, LongWritable> {
    //  根据前面的Mapper步骤，可能收到的键值对有：
    //  1.<类别，词数>
    //  2.<类别#词，次数>
    private MultipleOutputs<Text, LongWritable> multiout = null;


    @Override
    public void setup(Context context)throws IOException, InterruptedException{
        multiout = new MultipleOutputs<>(context);
    }

    @Override
    public void reduce(Text key, Iterable<Text> values, Context context)
    throws IOException, InterruptedException
    {
        String outPath = null;
        long sum = 0;
        String keyStr = key.toString();
        if(keyStr.split("#").length == 2){
            //收到了第二种情况，将文件其输出到另外属性值频度的文件
            outPath = "attrCount/";
        }
        else{
            //收到了第一种情况，将文件输出到另一个文件夹
            outPath = "classCount/";
        }
        for(Text val: values){
            sum += Long.parseLong(val.toString());
        }
        multiout.write(key, new LongWritable(sum), outPath + "info");
    }

    @Override public void cleanup(Reducer.Context context)throws IOException, InterruptedException{
        if(null != multiout){
            multiout.close();
            multiout = null;
        }
    }

}
