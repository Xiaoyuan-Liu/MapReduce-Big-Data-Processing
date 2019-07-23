package Vectorize;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class VectorizeMapper extends Mapper<Object, Text, Text, Text> {

    private List<String> wordList = new ArrayList<String>();
    //private List<Integer> wordCountList = new ArrayList<Integer>();//词出现的总次数

    public void setup(Context context)
            throws IOException, InterruptedException {
        /*超参数中存放着WordList文件的路径，下面这个函数用于读取该路径中的所有文件，理论上应该只有一个文件*/
        Path[] allFile = FilePath.getFolders(context.getProfileParams(), context.getConfiguration());
        long totalWord = 0;
        long wordType = 0;
        for(Path p: allFile){
            FSDataInputStream inStream = FileSystem.get(context.getConfiguration()).open(p);
            String line;
            while(inStream.available() > 0){
                line = inStream.readLine();
                String[] wordAndCnt = line.split("\t");
                String word = wordAndCnt[0];
                wordList.add(word);
                totalWord += Integer.parseInt(wordAndCnt[1]);
                wordType += 1;
                //wordCountList.add(Integer.parseInt(wordAndCnt[1]));
            }
        }
        context.write(new Text("!"), new Text(Long.toString(totalWord) +":" + Long.toString(wordType)));
    }

    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {


        FileSplit fileSplit = (FileSplit) context.getInputSplit();
        String fileName = fileSplit.getPath().getName();//得到文件名 文件名形式为class#filename，用#分隔出原名和类别
        String fileType = fileSplit.getPath().getParent().getName();
        String line = value.toString();


        //首先取得词和TF值
        String word = line.split("\t")[0];
        int wordTF = Integer.parseInt(line.split("\t")[1]);

        //取得词在IDF表中的索引
        int wordIndex = wordList.indexOf(word);


        //发射<词所在文件，词#词在单个文件中的count值>。
        if (wordIndex > -1)
            context.write(new Text(fileType + "/" + fileName), new Text(wordIndex + "#" + wordTF));

    }
}

