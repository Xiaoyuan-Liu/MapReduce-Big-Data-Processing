package Count;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class CountMapper extends Mapper<Object, Text, Text, IntWritable> {

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

    @Override
    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        //输入key是文件行，value是行的内容
        //输出是<文件类/文件名#单词，1>

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

            //然后去除停用词,停用词包括数字和一些常见词汇，发射<文件类/文件名#词，次数1>
            if ((!stopwords.contains(temp)) && (!temp.equals(""))) {
                Text word = new Text();
                word.set(fileType + "/" + fileName + "#" + temp);
                context.write(word, new IntWritable(1));
            }
        }
    }

    /*public void cleanup(Context context)
        //把总次数发射出去，并且设置内容为叹号，在ASCII表中，叹号比任何字母都靠前，因为在Combine的时候第一个处理的一定是叹号
            throws IOException, InterruptedException {
        context.write(new Text(fileType + "/" + fileName + "#" + "!"), new DoubleWritable(totalWord));
    }*/
}

