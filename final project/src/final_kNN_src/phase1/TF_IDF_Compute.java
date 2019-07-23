
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.DoubleWritable;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.fs.FileSystem;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.net.URI;

//Hadoop计算文档的TF-IDF向量，在Map阶段，将文档名 单词 和1发射出去。Combiner加起来
public class TF_IDF_Compute {

    //-----------------------Job1:Compute TF----------------------------


    public static class TFOutputFormat
            extends TextOutputFormat<Text, NullWritable> {
        @Override
        public void checkOutputSpecs(JobContext jobContext) throws IOException, FileAlreadyExistsException {
            return;
        }
    }

    //Mapper 发射<文件名#词,词数>
    public static class TFMapper
            extends Mapper<Object, Text, Text, DoubleWritable> {
        private int totalWord = 0;
        private String fileName = "";
        private String fileType = "";   //表示文件是训练集还是测试集
        private Set<String> stopwords;
        private Set<String> stopPunc;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            /*读取停用词表和标点符号*/
            //停用词表。stopwords是英文停用词，stopPuncNum是去除所有的标点符号和数字
            URI[] cacheFile = context.getCacheFiles();
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Path stopwordPath = new Path(cacheFile[0]);
            FSDataInputStream inStream = fs.open(stopwordPath);
            stopwords = new HashSet<String>();
            String line;
            while (inStream.available() > 0) {
                /*使用readLine是因为使用readUTF()会出现EOFException*/
                line = inStream.readLine();
                stopwords.add(line);
            }
            Path stopPuncPath = new Path(cacheFile[1]);
            FSDataInputStream inStream2 = fs.open(stopPuncPath);
            stopPunc = new HashSet<String>();
            while (inStream2.available() > 0) {
                line = inStream2.readLine();
                stopPunc.add(line);
            }
        }

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            //输入key是文件行，value是行的内容
            //输出是<文件名#单词，1>

            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            fileName = fileSplit.getPath().getName();//得到文件名
            fileType = fileSplit.getPath().getParent().getParent().getName();
            String line = value.toString().toLowerCase();

            //首先去除句子中所有的标点
            for (String str : stopPunc) {
                line = line.replace(str, " ");
            }

            //Text word = new Text();
            //Text fileName_lineOffset = new Text(fileName +"#"+key.toString());
            StringTokenizer itr = new StringTokenizer(line);
            for (; itr.hasMoreTokens(); ) {
                String temp = itr.nextToken();

                //然后去除停用词,停用词包括数字和一些常见词汇，发射<词#文件名，次数1>
                if ((!stopwords.contains(temp)) && (!temp.equals(""))) {
                    totalWord++;
                    Text word = new Text();
                    word.set(fileType + "/" + fileName + "#" + temp);
                    context.write(word, new DoubleWritable(1));
                }
            }
        }

        public void cleanup(Context context)
            //把总次数发射出去，并且设置内容为叹号，在ASCII表中，叹号比任何字母都靠前，因为在Combine的时候第一个处理的一定是叹号
                throws IOException, InterruptedException {
            context.write(new Text(fileType + "/" + fileName + "#" + "!"), new DoubleWritable(totalWord));
        }
    }

    //Combiner将同一个词中多次出现的词次数进行累加，然后除以文档总词数，得到词频TF。之所以Mapper输出采用FloatWritable的原因是，Combiner的输出类型必须与Mapper一致
    public static class TFCombiner
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        //输入是<文件名#单词，1>
        //输出是<文件名#单词，词频>
        private DoubleWritable result = new DoubleWritable();
        private double totalWord = 0;

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            //int splitSymbolIndex = key.toString().indexOf("#");
            if (key.toString().split("#")[1].equals("!")) {
                //取得总词数，然后抛弃这对key-value
                for (DoubleWritable val : values)
                    totalWord += val.get();
            } else {
                //计算词频除以总词数，即TF值
                double tf = 0;
                for (DoubleWritable val : values) {
                    tf += val.get();
                }
                tf = tf / totalWord;
                if (!Double.isInfinite(tf)) {
                    result.set(tf);
                    context.write(key, result);
                }
            }
        }
    }


    public static class TFPartitioner
            extends HashPartitioner<Text, DoubleWritable> {
        //输入是<文件名#单词，词频>
        //输出是以<文件名,词频>划分的Reduce服务器号
        public int getPartition(Text key, DoubleWritable value, int numReduceTasks) {
            String term = new String();
            term = key.toString().split("#")[0];
            //System.err.println(term);
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    public static class TFReducer
            extends Reducer<Text, DoubleWritable, NullWritable, Text> {
        private Text word1 = new Text();
        private Text word2 = new Text();
        String temp = new String();
        static Text CurrentItem = new Text(" ");
        static List<String> postingList = new ArrayList<String>();

        //使用MUltipleOutputs类，输出为多文件
        private MultipleOutputs<NullWritable, Text> multipleOutputs = null;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            multipleOutputs = new MultipleOutputs<NullWritable, Text>(context);
        }

        //reduce输出的key-value对格式:term <doc1,num1>...<total,sum>
        @Override
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            String fileName = key.toString().split("#")[0];
            String filePath = fileName.split("/")[0];
            fileName = fileName.split("/")[1];
            String word = key.toString().split("#")[1];
            //System.err.println(key.toString());
            DecimalFormat df = new DecimalFormat();
            df.setMaximumFractionDigits(10);
            for (DoubleWritable val : values)
                //输出文件名为“类别#原文件名”，每行为一个单词及其TF值，用制表符隔开
                multipleOutputs.write(NullWritable.get(), new Text(word + "\t" + df.format(val.get())), filePath+ "/" + context.getProfileParams() + "#" + fileName);
        }

        public void cleanup(Reducer.Context context)
                throws IOException, InterruptedException {
            if (null != multipleOutputs) {
                multipleOutputs.close();
                multipleOutputs = null;
            }
        }
    }


    //-------------------------------Job2:Compute IDF----------------------------------
    //计算IDF值。从TF步骤的输出文件夹中，统计所有词的
    public static class IDFMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        public void map(Object key, Text value, Context context)

            //map读取TF步骤的所有文件，遇到一个词就发射<词，次数1>。不需要Combine操作，因为TF生成的文件中每个词必定只出现了一次
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            Text word = new Text(value.toString().split("\t")[0]);
            DoubleWritable tf = new DoubleWritable(Double.parseDouble(value.toString().split("\t")[1]));
            context.write(new Text(word), new IntWritable(1));
        }

    }

    public static class IDFReducer
            extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        private int fileTotal = 0;

        public void setup(Context context)
                throws IOException, InterruptedException {

            //从超参数中取得文件总数
            fileTotal = Integer.parseInt(context.getProfileParams());
        }

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            //每个词出现的总词数除以总文件数就是IDF值
            double wordTotal = 0;
            for (IntWritable val : values) {
                wordTotal += val.get();
            }
            wordTotal = fileTotal / wordTotal;
            if (!Double.isInfinite(wordTotal))
                context.write(new Text(key.toString()), new DoubleWritable(wordTotal));
        }

    }


    //-----------------------------------Job3:Compute TF-IDF------------------------------


    public static class TFIDFMapper
            extends Mapper<Object, Text, Text, Text> {

        List<String> wordList = new ArrayList<String>();
        List<Double> wordIDFList = new ArrayList<Double>();


        public void setup(Context context)
                throws IOException, InterruptedException {
            /*超参数中存放着IDF文件的路径，下面这个函数用于读取该路径中的所有文件，理论上应该只有一个文件*/
            Path[] allFile = FilePath.getFolders(context.getProfileParams(), context.getConfiguration());
            if(allFile.length > 1){
                System.err.println("Files too many!");
                System.err.print("File number:");
                System.err.println(allFile.length);
                System.exit(-1);
            }
            for(Path p: allFile){
                FSDataInputStream inStream = FileSystem.get(context.getConfiguration()).open(p);
                String line;
                while(inStream.available() > 0){
                    line = inStream.readLine();
                    String[] wordAndIDF = line.split("\t");
                    String word = wordAndIDF[0];
                    wordList.add(word);
                    wordIDFList.add(Double.parseDouble(wordAndIDF[1]));
                }
            }
        }

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {


            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName();//得到文件名 文件名形式为class#filename，用#分隔出原名和类别
            String fileType = fileSplit.getPath().getParent().getName();
            String line = value.toString();


            //首先取得词和TF值
            String word = line.split("\t")[0];
            double wordTF = Double.parseDouble(line.split("\t")[1]);

            //取得词在IDF表中的索引
            int wordIndex = wordList.indexOf(word);

            //避免以科学计数法输出
            DecimalFormat df = new DecimalFormat();
            df.setMaximumFractionDigits(10);

            //发射<词所在文件，词#词的TFIDF值>。
            if (wordIndex > -1)
                context.write(new Text(fileType + "/" + fileName), new Text(wordIndex + "#" + df.format(wordTF * wordIDFList.get(wordIndex))));

        }
    }

    /*
    public static class TFIDFPartitioner
            extends HashPartitioner<Text, DoubleWritable> {
        //输入是<文件名#单词，TFIDF值>
        //输出是以<文件名,TFIDF值>划分的Reduce服务器号
        public int getPartition(Text key, DoubleWritable value, int numReduceTasks) {
            String term = new String();
            term = key.toString().split("#")[0];
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }*/


    public static class TFIDFReducer
            extends Reducer<Text, Text, NullWritable, Text> {

        String fileName = null;
        StringBuilder vector = new StringBuilder();
        //reduce操作，还是先从IDF文件中读取所有单词，然后把TFIDF值表初始化为0.然后在reduce操作中，将map发射的TFIDF值写到TFIDF表上去，然后将所有TFIDF值合并成一个向量，输出为一行
        List<String> wordList = new ArrayList<String>();
        List<String> wordTFIDFList = new ArrayList<String>();
        //使用MUltipleOutputs类，输出为多文件
        private MultipleOutputs<NullWritable, Text> multipleOutputs = null;
        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            multipleOutputs = new MultipleOutputs<NullWritable, Text>(context);
            /*Path[] allFile = FilePath.getFolders(context.getProfileParams(), context.getConfiguration());
            if(allFile.length > 1){
                System.err.println("Files too many!");
                System.err.print("File number:");
                System.err.println(allFile.length);
                System.exit(-1);
            }
            FSDataInputStream inStream = FileSystem.get(context.getConfiguration()).open(allFile[0]);
            String line;
            while(inStream.available() > 0){
                line = inStream.readLine();
                String[] wordAndIDF = line.split("\t");
                String[] numberAndWord = wordAndIDF[0].split(":");
                wordList.add(numberAndWord[1]);
            }*/
        }


        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            //String fileName = key.toString().split("#")[0];
            //String word = key.toString().split("#")[1];
            //System.err.println(key.toString());


            //将邮件中出现的词，将其TFIDF值写入TFIDF表
            if(fileName == null){
                fileName = key.toString();
                String type = fileName.split("/")[0];
                String name = fileName.split("/")[1];
                if(type.equals("20_newsgroup"))
                    vector.append(name.split("#")[0] + "\t");
                else
                    vector.append(name + "\t");
            }
            if(!fileName.equals(key.toString())) {
                String type = fileName.split("/")[0];
                if(type.equals("20_newsgroup"))
                    multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "train/trainVec");
                else
                    multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "test/testVec");
                //context.write(NullWritable.get(), new Text(vector.toString()));
                vector = new StringBuilder();
                fileName = key.toString();
                type = fileName.split("/")[0];
                String name = fileName.split("/")[1];
                if(type.equals("20_newsgroup"))
                    vector.append(name.split("#")[0] + "\t");
                else
                    vector.append(name + "\t");
            }
            for (Text val : values) {
                String[] vals = val.toString().split("#");
                vector.append(vals[0] + ":" + vals[1] + " ");
            }
        }


        public void cleanup(Reducer.Context context)
                throws IOException, InterruptedException {
            String type = fileName.split("/")[0];
            if(type.equals("20_newsgroup"))
                multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "train/trainVec");
            else
                multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "test/testVec");
            //context.write(NullWritable.get(), new Text(vector.toString()));

            if (null != multipleOutputs) {
                multipleOutputs.close();
                multipleOutputs = null;
            }
        }
    }


    //此函数用于从文件系统中取得Job1，即计算TF值产生的所有文件
    public static int getFileNum(String folderPath, Configuration conf) throws IOException {
        FileSystem hdfs = FileSystem.get(URI.create(folderPath), conf);
        FileStatus[] fs = hdfs.listStatus(new Path(folderPath));
        Path[] subFolder = FileUtil.stat2Paths(fs);
        int fileCount = 0;
        for (Path p : subFolder) {
            fileCount += hdfs.listStatus(p).length;
        }
        return fileCount;

        /*File f = new File(folderPath);
        if (f.exists() && f.isDirectory()) {
            int count = f.list(new FilenameFilter() {
                public boolean accept(File dir, String name) {
                    if (name.contains("_SUCCESS")||name.contains(".crc"))
                        return false;
                    else
                        return true;
                }
            }).length;
            return count;
        } else return 0;*/
    }


    /*输入一共有四个参数
     * 第一个参数是输入训练集文件夹
     * 第二个参数是输入测试集文件夹
     * 第三个参数是停词表的文件
     * 第四个参数是标点符号的文件夹
     *
     *
     * 输出的内容有：
     * 第一阶段输出所有词的TF，存放在 /out/TF-IDF/TF/ 中
     * 第二阶段输出所有词的IDF，存放在 /out/TF-IDF/IDF/ 中
     * 第三阶段输出每个样本中的TF-IDF， 每一行代表一个文本，第一个字符串是类别，而后面就是每个词的TF-IDF值
     *冒号前是该词的编号，冒号后是该词的TF-IDF值。
     * 第四阶段即将开始进行KNN分类，因此前面在计算TF-IDF时，也要考虑对测试集进行向量化
     * */
    public static void main(String[] args)
            throws Exception {

        String p2Folder = "/tmp/2019st40/out/TF-IDF/";

        //检查输入参数数量
        if (args.length != 4) {
            System.err.println("Input parameters not enough");
            System.exit(2);
        }


        //检查输入目录是否存在

        if (true) {
            Configuration conf = new Configuration();
            /*先计算训练集的TF*/
            Path[] trainPathList = FilePath.getFolders(args[0], conf);
            Path[] testPathList = FilePath.getFolders(args[1], conf);
            Path[] pathList = (Path[])ArrayUtils.addAll(trainPathList, testPathList);
            for (Path classPath : pathList) {
                Job job1 = Job.getInstance(conf, "Compute TF");
                job1.setJarByClass(TF_IDF_Compute.class);
                job1.addCacheFile(new Path(args[2]).toUri());
                job1.addCacheFile(new Path(args[3]).toUri());
                job1.setMapperClass(TFMapper.class);
                job1.setCombinerClass(TFCombiner.class);
                job1.setPartitionerClass(TFPartitioner.class);
                job1.setReducerClass(TFReducer.class);

                job1.setMapOutputKeyClass(Text.class);
                job1.setMapOutputValueClass(DoubleWritable.class);

                job1.setOutputKeyClass(Text.class);
                job1.setOutputValueClass(Text.class);
                String[] classPathSplit = classPath.toString().split("/");
                job1.setProfileParams(classPathSplit[classPathSplit.length - 1]);
                job1.setOutputFormatClass(TFOutputFormat.class);
                FileInputFormat.addInputPath(job1, classPath);
                FileOutputFormat.setOutputPath(job1, new Path(p2Folder + "TF/"));
                job1.waitForCompletion(true);
            }



            Configuration conf2 = new Configuration();
            Job job2 = Job.getInstance(conf2, "Compute IDF");
            job2.setJarByClass(TF_IDF_Compute.class);

            job2.setMapperClass(IDFMapper.class);
            job2.setReducerClass(IDFReducer.class);

            job2.setMapOutputKeyClass(Text.class);
            job2.setMapOutputValueClass(IntWritable.class);

            job2.setOutputKeyClass(Text.class);
            job2.setOutputValueClass(DoubleWritable.class);
            Path outPath2 = new Path(p2Folder + "IDF/");
            FileSystem fileSystem2 = outPath2.getFileSystem(conf2);
            /*在进行第二次任务前，要删除前一个文件产生的 part-r-00000、以及_SUCCESS */
            fileSystem2.delete(new Path(p2Folder + "TF/part-r-00000"), false);
            //fileSystem2.delete(new Path("/out/TF-IDF/TF/train/part-r-00000"), false);
            fileSystem2.delete(new Path(p2Folder+ "TF/_SUCCESS"), false);
            //fileSystem2.delete(new Path("/out/TF-IDF/TF/train/_SUCCESS"), false);

            //设置超参数：文件总数，用以计算IDF值
            job2.setProfileParams(Integer.toString(getFileNum(p2Folder + "TF/20_newsgroup/", conf2)));
            FileInputFormat.addInputPath(job2, new Path(p2Folder + "TF/20_newsgroup/"));
            FileOutputFormat.setOutputPath(job2, new Path(p2Folder + "IDF/"));
            job2.waitForCompletion(true);
            /*删除job2产生的多余的文件*/



            Configuration conf3 = new Configuration();
            Job job3 = Job.getInstance(conf3, "Compute TFIDF");
            FileSystem fs = outPath2.getFileSystem(conf3);
            fs.delete(new Path(p2Folder + "/IDF/_SUCCESS"), false);
            job3.setJarByClass(TF_IDF_Compute.class);

            job3.setMapperClass(TFIDFMapper.class);
            //job3.setPartitionerClass(TFIDFPartitioner.class);
            job3.setReducerClass(TFIDFReducer.class);

            job3.setMapOutputKeyClass(Text.class);
            job3.setMapOutputValueClass(Text.class);

            job3.setOutputKeyClass(Text.class);
            job3.setOutputValueClass(Text.class);
            job3.setProfileParams(p2Folder + "IDF/");
            FileInputFormat.addInputPath(job3, new Path(p2Folder + "TF/20_newsgroup/"));
            FileInputFormat.addInputPath(job3, new Path(p2Folder + "TF/TestDataSet/"));
            FileOutputFormat.setOutputPath(job3, new Path(p2Folder + "TF-IDF/"));

            job3.waitForCompletion(true);

        } else {
            System.err.println(args[0]);
            System.exit(-2);


        }
    }
}

        /*File f2 = new File(args[4]);
        if (f2.exists()) {
            //Path outPath = new Path(args[1]);
            //Configuration tmpConf = new Configuration();
            //FileSystem fileSystem = outPath.getFileSystem(tmpConf);
            //if (fileSystem.exists(outPath)) {
            //    fileSystem.delete(outPath, true);
            //}

            String[] fileList = f2.list();

            for (String className : fileList) {

                Configuration conf = new Configuration();
                Job job1 = Job.getInstance(conf, "Compute TF");
                job1.setJarByClass(TF_IDF_Compute.class);

                job1.setMapperClass(TFMapper.class);
                job1.setCombinerClass(TFCombiner.class);
                job1.setPartitionerClass(TFPartitioner.class);
                job1.setReducerClass(TFReducer.class);

                job1.setMapOutputKeyClass(Text.class);
                job1.setMapOutputValueClass(DoubleWritable.class);

                job1.setOutputKeyClass(Text.class);
                job1.setOutputValueClass(Text.class);

                job1.setProfileParams(className);
                job1.setOutputFormatClass(TFOutputFormat.class);
                FileInputFormat.addInputPath(job1, new Path(args[4] + className));
                FileOutputFormat.setOutputPath(job1, new Path(args[5]));
                job1.waitForCompletion(true);


            }



            Configuration conf3 = new Configuration();
            Job job3 = Job.getInstance(conf3, "Compute TFIDF");
            job3.setJarByClass(TF_IDF_Compute.class);

            job3.setMapperClass(TFIDFMapper.class);
            //job3.setPartitionerClass(TFIDFPartitioner.class);
            job3.setReducerClass(TFIDFReducer.class);

            job3.setMapOutputKeyClass(Text.class);
            job3.setMapOutputValueClass(Text.class);

            job3.setOutputKeyClass(Text.class);
            job3.setOutputValueClass(Text.class);
            Path outPath3 = new Path(args[6]);
            FileSystem fileSystem3 = outPath3.getFileSystem(conf3);
            if (fileSystem3.exists(outPath3)) {
                fileSystem3.delete(outPath3, true);
            }
            job3.setProfileParams(args[2]);
            FileInputFormat.addInputPath(job3, new Path(args[5]));
            FileOutputFormat.setOutputPath(job3, new Path(args[6]));

            job3.waitForCompletion(true);





            Configuration conf4 = new Configuration();
            Job job4 = Job.getInstance(conf4, "KNNPreparation");
            job4.setJarByClass(TF_IDF_Compute.class);

            job4.setMapperClass(KNNPrepMapper.class);
            //job3.setPartitionerClass(TFIDFPartitioner.class);
            job4.setReducerClass(KNNPrepReducer.class);

            job4.setMapOutputKeyClass(Text.class);
            job4.setMapOutputValueClass(Text.class);

            job4.setOutputKeyClass(Text.class);
            job4.setOutputValueClass(Text.class);
            Path outPath4 = new Path(args[7]);
            FileSystem fileSystem4 = outPath4.getFileSystem(conf4);
            if (fileSystem4.exists(outPath4)) {
                fileSystem4.delete(outPath4, true);
            }
            FileInputFormat.addInputPath(job4, new Path(args[6]));
            FileOutputFormat.setOutputPath(job4, new Path(args[7]));

            job4.waitForCompletion(true);


            Configuration conf5 = new Configuration();
            Job job5 = Job.getInstance(conf5, "KNN");
            job5.setJarByClass(TF_IDF_Compute.class);

            job5.setMapperClass(KNNMapper.class);
            //job3.setPartitionerClass(TFIDFPartitioner.class);
            //job5.setReducerClass(TFIDFReducer.class);

            job5.setMapOutputKeyClass(Text.class);
            job5.setMapOutputValueClass(Text.class);

            //job5.setOutputKeyClass(Text.class);
            //job5.setOutputValueClass(Text.class);
            Path outPath5 = new Path(args[8]);
            FileSystem fileSystem5 = outPath5.getFileSystem(conf5);
            if (fileSystem5.exists(outPath5)) {
                fileSystem5.delete(outPath5, true);
            }
            job5.setProfileParams(args[3]);
            FileInputFormat.addInputPath(job5, new Path(args[7]));
            FileOutputFormat.setOutputPath(job5, new Path(args[8]));

            System.exit(job5.waitForCompletion(true) ? 0 : 1);


        } else {
            System.err.println(args[0]);
            System.exit(-2);


        }
    }






    //-------------------------------Job5:KNN--------------------------------
    //输入文件是待预测邮件。超参数设置为训练集目录，训练集名称为“类别#文件名”

    public static class KNNMapper
    extends Mapper<Object, Text, Text, Text>{

        //读入训练集
        private ArrayList<ArrayList<String>> trainVectors = new ArrayList<ArrayList<String>>();
        //setup中，取得超参数，将文件读取并储存在内存中，文件内容用字符串数组储存，字符串数组首为类别，之后储存训练集文档向量
        public void setup(Context context)
        throws IOException, InterruptedException{
            String fileName = context.getProfileParams();//取得超参数
            File f = new File(fileName);
            Path TraingPath = null;
            if(f.exists()){
                String[] fileList = f.list(new FilenameFilter() {
                    public boolean accept(File dir, String name) {
                        if (name.contains("_SUCCESS")||name.contains(".crc"))
                            return false;
                        else
                            return true;
                    }
                });
                //TraingPath = new Path(fileName+fileList[0]);
                for(String str:fileList){
                    TraingPath = new Path(fileName+str);
                    String line = null;
                    BufferedReader br = new BufferedReader(new FileReader(TraingPath.toString()));

                    if((line = br.readLine()) != null){
                        String[] vector = line.split("\t");
                        ArrayList<String> classVector = new ArrayList<String>();
                        classVector.add(str.split("#")[0]);
                        for(String element:vector)
                            classVector.add(element);
                        trainVectors.add(classVector);
                    }

                }
            }else{
                System.err.println("Training Files not Exist");
                System.exit(-3);
            }
        }


        //所有测试样本都放在一个文件中。目前K设为10
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName();//得到文件名

            if (fileName.contains("_SUCC") || fileName.contains(".crc")) {
                Double.parseDouble(fileName);
                return;
            } else {
                String[] testVector = value.toString().split("\t");//注意，字符串数组索引0的位置是文件名
                double[][] KNN = new double[10][2];//储存最近邻，每行储存距离和该点在训练集中的索引
                for (double[] vector : KNN) {
                    vector[0] = Double.POSITIVE_INFINITY;
                    vector[1] = -1;
                }
                for (int index = 0; index < trainVectors.size(); index++) {
                    double distance = 0;

                    //计算余弦相似度
                    for (int i = 1; i < trainVectors.get(index).size(); i++)
                        distance += Double.parseDouble((String) trainVectors.get(index).get(i)) * Double.parseDouble(testVector[i]);
                    int replaceIndex = -1;
                    for (int i = 0; i < 10; i++) {
                        if (KNN[i][0] > distance) {
                            //如果刚开始寻找，或者新找到的点还要原先寻找到的点还要远
                            if ((replaceIndex == -1) || (KNN[i][0] > KNN[replaceIndex][0])) {
                                replaceIndex = i;
                            }
                        }
                    }
                    if (replaceIndex > -1) {
                        KNN[replaceIndex][0] = distance;
                        KNN[replaceIndex][1] = index;
                    }
                }
                ArrayList<String> className = new ArrayList<String>();
                ArrayList<Integer> classCount = new ArrayList<Integer>();

                for (int i = 0; i < 10; i++) {
                    if (className.contains(trainVectors.get((int) KNN[i][1]).get(0))) {
                        int index = className.indexOf(trainVectors.get((int) KNN[i][1]).get(0));
                        classCount.set(index, classCount.get(index) + 1);
                    } else {
                        className.add(trainVectors.get((int) KNN[i][0]).get(0));
                        classCount.add(new Integer(1));
                    }
                }


                //寻找类别最多的

                int index = -1;

                for (int i = 0; i < classCount.size(); i++) {
                    if ((index == -1) || (classCount.get(i) > classCount.get(index))) {
                        index = i;
                    }
                }

                context.write(new Text(testVector[0]), new Text(className.get(index)));

            }
        }
    }


    //---------------------job4:KNN preparation---------------------
    public static class KNNPrepMapper
            extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context)

            //map读取TF步骤的所有文件，遇到一个词就发射<词，次数1>。不需要Combine操作，因为TF生成的文件中每个词必定只出现了一次
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName();//得到文件名

            context.write(new Text(fileName), value);
        }

    }

    public static class KNNPrepReducer
            extends Reducer<Text, Text, Text, Text> {
        private int fileTotal = 0;

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for(Text val:values)
                context.write(key, val);
        }
    }
}*/
