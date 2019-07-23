import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileAlreadyExistsException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.regex.Pattern;

public class Classifier {
    /*重写该接口避免出现重复输出文件夹的错误*/
    public static class TFOutputFormat
            extends TextOutputFormat<Text, NullWritable> {
        @Override
        public void checkOutputSpecs(JobContext jobContext) throws IOException, FileAlreadyExistsException {
            return;
        }
    }

    public static void main(String args[])
        throws Exception{

        Pattern pattern= Pattern.compile("[0-9]+");
        if(!pattern.matcher(args[0]).matches()){
            System.err.println("the neighbor number in kNN must be integer!");
            System.exit(-1);
        }
        Configuration conf = new Configuration();
        Path testSetPath = new Path("/tmp/2019st40/out/TF-IDF/TF-IDF/test/");
        Path trainSetPath = new Path("/tmp/2019st40/out/TF-IDF/TF-IDF/train/");

        Job clsf = Job.getInstance(conf, "classify");
        clsf.setJarByClass(Classifier.class);
        clsf.addCacheFile(trainSetPath.toUri());
        clsf.setMapperClass(KNNMapper.class);
        clsf.setProfileParams(args[0]);
        clsf.setMapOutputKeyClass(Text.class);
        clsf.setMapOutputValueClass(Text.class);
        clsf.setProfileParams(args[0]);

        FileInputFormat.addInputPath(clsf, testSetPath);
        FileOutputFormat.setOutputPath(clsf, new Path("/tmp/2019st40/out/result/"));


        Job accuracy = Job.getInstance(conf);
        accuracy.setJarByClass(Classifier.class);
        accuracy.setMapperClass(AccuracyMapper.class);
        accuracy.setCombinerClass(AccuracyCombiner.class);
        accuracy.setReducerClass(AccuracyReducer.class);
        accuracy.setMapOutputValueClass(DoubleWritable.class);
        accuracy.setMapOutputKeyClass(Text.class);
        accuracy.setOutputKeyClass(Text.class);
        accuracy.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(accuracy, new Path("/tmp/2019st40/out/result/"));
        FileOutputFormat.setOutputPath(accuracy, new Path("/tmp/2019st40/out/accuracy/"));

        if(clsf.waitForCompletion(true)){
            /*分类完成后删除多余的_SUCCESS文件*/
            FileSystem fs = testSetPath.getFileSystem(conf);
            fs.delete(new Path("/tmp/2019st40/out/result/_SUCCESS"), false);
            System.exit(accuracy.waitForCompletion(true)?0:1);
        }
    }








}



