package Vectorize;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class VectorizeReducer extends Reducer<Text, Text, NullWritable, Text> {
    String fileName = null;
    StringBuilder vector = new StringBuilder();
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
        if(key.toString().equals("!")){
            //把记录着总词数的记录单独列出
            Text info = null;
            for(Text val : values){
                info = val;
            }
            context.write(NullWritable.get(), info);
        }
        else {
            if (fileName == null) {
                fileName = key.toString();
                String type = fileName.split("/")[0];
                String name = fileName.split("/")[1];
                if (type.equals("20_newsgroup"))
                    vector.append(name.split("#")[0] + "\t");
                else
                    vector.append(name + "\t");
            }
            if (!fileName.equals(key.toString())) {
                String type = fileName.split("/")[0];
                if (type.equals("20_newsgroup"))
                    multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "train/trainVec");
                else
                    multipleOutputs.write(NullWritable.get(), new Text(vector.toString()), "test/testVec");
                //context.write(NullWritable.get(), new Text(vector.toString()));
                vector = new StringBuilder();
                fileName = key.toString();
                type = fileName.split("/")[0];
                String name = fileName.split("/")[1];
                if (type.equals("20_newsgroup"))
                    vector.append(name.split("#")[0] + "\t");
                else
                    vector.append(name + "\t");
            }
            for (Text val : values) {
                String[] vals = val.toString().split("#");
                vector.append(vals[0] + ":" + vals[1] + " ");
            }
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
