import Accuracy.*;

import Predict.PredictMapper;
import TrainModel.TrainModelMapper;
import TrainModel.TrainModelReducer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;


/*使用朴素贝叶斯进行预测，采用的多项式分布来进行计算*/
public class NB {

    public static void main(String args[])
    throws Exception {

        Configuration conf = new Configuration();
        Path testSetPath = new Path("/tmp/2019st40/out2/CountVector/Vector/test/");
        Path trainSetPath = new Path("/tmp/2019st40/out2/CountVector/Vector/train");
        Path modelPath = new Path("/tmp/2019st40/out2/model/");
        Path resultPath = new Path("/tmp/2019st40/out2/result/");

        Job buildModel = Job.getInstance(conf, "NB");
        buildModel.setJarByClass(NB.class);
        buildModel.setMapperClass(TrainModelMapper.class);
        buildModel.setReducerClass(TrainModelReducer.class);
        buildModel.setMapOutputKeyClass(Text.class);
        buildModel.setMapOutputValueClass(Text.class);
        buildModel.setOutputKeyClass(Text.class);
        buildModel.setOutputValueClass(LongWritable.class);

        FileInputFormat.addInputPath(buildModel, trainSetPath);
        FileOutputFormat.setOutputPath(buildModel, modelPath);


        Job predict = Job.getInstance(conf);
        predict.setJarByClass(NB.class);
        predict.setMapperClass(PredictMapper.class);
        predict.setMapOutputKeyClass(Text.class);
        predict.setMapOutputValueClass(Text.class);
        predict.addCacheFile(new Path("/tmp/2019st40/out2/CountVector/Vector/part-r-00000").toUri());
        predict.addCacheFile(new Path(modelPath.toString() + "/classCount/info-r-00000").toUri());
        predict.addCacheFile(new Path(modelPath.toString() + "/attrCount/info-r-00000").toUri());

        FileInputFormat.addInputPath(predict, testSetPath);
        FileOutputFormat.setOutputPath(predict, resultPath);

        Job accuracy = Job.getInstance(conf);
        accuracy.setJarByClass(NB.class);
        accuracy.setMapperClass(AccuracyMapper.class);
        accuracy.setCombinerClass(AccuracyCombiner.class);
        accuracy.setReducerClass(AccuracyReducer.class);
        accuracy.setMapOutputValueClass(DoubleWritable.class);
        accuracy.setMapOutputKeyClass(Text.class);
        accuracy.setOutputKeyClass(Text.class);
        accuracy.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(accuracy, new Path("/tmp/2019st40/out2/result/"));
        FileOutputFormat.setOutputPath(accuracy, new Path("/tmp/2019st40/out2/accuracy/"));

        if(buildModel.waitForCompletion(true)){
            if (predict.waitForCompletion(true)) {
                System.exit(accuracy.waitForCompletion(true)?0:1);
            }
        }
        System.exit(-1);

    }
}
