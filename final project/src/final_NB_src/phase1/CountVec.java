import Count.*;

import ListWords.ListWordsMapper;
import ListWords.ListWordsReducer;

import Vectorize.FilePath;

import Vectorize.VectorizeMapper;
import Vectorize.VectorizeReducer;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileAlreadyExistsException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

public class CountVec {
    /*用来避免出现输入输出错误：文件夹已存在*/
    public static class TFOutputFormat
            extends TextOutputFormat<Text, NullWritable> {
        @Override
        public void checkOutputSpecs(JobContext jobContext) throws IOException, FileAlreadyExistsException {
            return;
        }
    }


    /*朴素贝叶斯若是使用多项式分布，则需要使用one-hot来表示文本向量*/
    /*一共需要四个输入
    * 第一个输入是训练集文本
    * 第二个输入是测试集
    * 第三个输入是停词表
    * 第四个输入是标点符号*/




    public static void main(String args[])
        throws Exception {
        String v2Folder = "/tmp/2019st40/out2/CountVector/";

        //检查输入
        if (args.length != 4) {
            System.err.println("Input parameters not enough");
            System.exit(2);
        }

        //开始将各个训练集和测试集中文档进行向量化
        Configuration conf = new Configuration();
        Path[] trainPathList = FilePath.getFolders(args[0], conf);
        Path[] testPathList = FilePath.getFolders(args[1], conf);
        Path[] pathList = (Path[]) ArrayUtils.addAll(trainPathList, testPathList);
        for (Path classPath : pathList) {
            Job countWord = Job.getInstance(conf, "Count word");
            countWord.setJarByClass(CountVec.class);
            countWord.addCacheFile(new Path(args[2]).toUri());
            countWord.addCacheFile(new Path(args[3]).toUri());
            countWord.setMapperClass(CountMapper.class);
            countWord.setCombinerClass(CountCombiner.class);
            countWord.setPartitionerClass(CountPartitioner.class);
            countWord.setReducerClass(CountReducer.class);

            countWord.setMapOutputKeyClass(Text.class);
            countWord.setMapOutputValueClass(IntWritable.class);

            countWord.setOutputKeyClass(NullWritable.class);
            countWord.setOutputValueClass(Text.class);
            String[] classPathSplit = classPath.toString().split("/");
            countWord.setProfileParams(classPathSplit[classPathSplit.length - 1]);
            countWord.setOutputFormatClass(TFOutputFormat.class);
            FileInputFormat.addInputPath(countWord, classPath);
            FileOutputFormat.setOutputPath(countWord, new Path(v2Folder + "Count/"));
            if (!countWord.waitForCompletion(true)) {
                System.err.println("count word errror while in file path: " + classPath.toString());
                System.exit(-1);
            }
        }
        Configuration conf2 = new Configuration();
        Job listWord = Job.getInstance(conf2, "ListWords");
        listWord.setJarByClass(CountVec.class);

        listWord.setMapperClass(ListWordsMapper.class);
        listWord.setReducerClass(ListWordsReducer.class);

        listWord.setMapOutputKeyClass(Text.class);
        listWord.setMapOutputValueClass(IntWritable.class);

        listWord.setOutputKeyClass(Text.class);
        listWord.setOutputValueClass(DoubleWritable.class);
        Path outPath2 = new Path(v2Folder + "WordsList/");
        FileSystem fileSystem2 = outPath2.getFileSystem(conf2);
        /*在进行第二次任务前，要删除前一个文件产生的 part-r-00000、以及_SUCCESS */
        fileSystem2.delete(new Path(v2Folder + "Count/part-r-00000"), false);
        //fileSystem2.delete(new Path("/out/TF-IDF/TF/train/part-r-00000"), false);
        fileSystem2.delete(new Path(v2Folder+ "Count/_SUCCESS"), false);

        FileInputFormat.addInputPath(listWord, new Path(v2Folder + "Count/20_newsgroup/"));
        FileOutputFormat.setOutputPath(listWord, new Path(v2Folder + "WordsList/"));
        if(!listWord.waitForCompletion(true)){
            System.err.println("List words error!");
            System.exit(-1);
        }

        Configuration conf3 = new Configuration();
        Job vectorize = Job.getInstance(conf3, "vectorzie");
        FileSystem fs = outPath2.getFileSystem(conf3);
        fs.delete(new Path(v2Folder + "/WordsList/_SUCCESS"), false);
        vectorize.setJarByClass(CountVec.class);

        vectorize.setMapperClass(VectorizeMapper.class);
        //vectorize.setPartitionerClass(TFIDFPartitioner.class);
        vectorize.setReducerClass(VectorizeReducer.class);

        vectorize.setMapOutputKeyClass(Text.class);
        vectorize.setMapOutputValueClass(Text.class);

        vectorize.setOutputKeyClass(Text.class);
        vectorize.setOutputValueClass(Text.class);
        vectorize.setProfileParams(v2Folder + "WordsList/");
        FileInputFormat.addInputPath(vectorize, new Path(v2Folder + "Count/20_newsgroup/"));
        FileInputFormat.addInputPath(vectorize, new Path(v2Folder + "Count/TestDataSet/"));
        FileOutputFormat.setOutputPath(vectorize, new Path(v2Folder + "Vector/"));

        System.exit(vectorize.waitForCompletion(true)?0 : 1);

    }
}
