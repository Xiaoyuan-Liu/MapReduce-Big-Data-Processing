package Predict;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;

import static java.lang.Math.*;

public class PredictMapper extends Mapper<Object, Text, Text, Text>{
    private HashMap<String, Long> FY = null;
    private HashMap<String, Long> FxY = null;
    private long totalWords = 0;
    private long wordTypes = 0;

    @Override
    public void setup(Context context) throws IOException, InterruptedException{
        //一共三个cachefile,第一个是存放总词数和词汇数的， 第二个是存放FY的，第三个是存放FxY的
        URI[] cacheFile = context.getCacheFiles();
        FileSystem fs = FileSystem.get(context.getConfiguration());
        Path wordsInfoPath = new Path(cacheFile[0]);

        FSDataInputStream inStream = fs.open(wordsInfoPath);
        String line = inStream.readLine();
        totalWords = Long.parseLong(line.split(":")[0]);
        wordTypes = Long.parseLong(line.split(":")[1]);

        FY = new HashMap<>();
        FxY = new HashMap<>();

        Path FYPath = new Path(cacheFile[1]);
        FSDataInputStream is2 = fs.open(FYPath);
        while(is2.available() > 0){
            line = is2.readLine();
            String[] temp = line.split("\t");
            FY.put(temp[0], Long.parseLong(temp[1]));
        }

        Path FxYPath = new Path(cacheFile[2]);
        FSDataInputStream is3 = fs.open(FxYPath);
        while(is3.available() > 0){
            line = is3.readLine();
            String[] temp = line.split("\t");
            FxY.put(temp[0], Long.parseLong(temp[1]));
        }

    }

    @Override
    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException{
        String[] oneSample = value.toString().split("\t");
        String sampleName = oneSample[0];
        String[] attributes = oneSample[1].split(" ");
        double maxP = Double.POSITIVE_INFINITY;
        String idx = new String("null");
        for(String cls : FY.keySet()){
            double tempP = 1.0;
            //Fcls是类Y的总词数
            long Fcls = FY.get(cls);
            double fm = Fcls + wordTypes;
            for(String attr: attributes){
                String attrName = attr.split(":")[0];
                String attrTime = attr.split(":")[1];
                String FxYkey = cls + "#" + attrName;
                double fz = 0;
                if(FxY.keySet().contains(FxYkey)){
                    fz = FxY.get(FxYkey) + 1.0;
                }
                else{
                    fz = 1.0;
                }
                //用对数是因为直接使用分数从而导致很容易出现概率为0，因为计算机的对小数的表达能力有限
                tempP = tempP * log(fz/fm);
            }
            double PreP = log((double)Fcls/(double)totalWords);
            tempP = abs(tempP * PreP);
            if(tempP < maxP){
                maxP = tempP;
                idx = cls;
            }
        }
        context.write(new Text(sampleName + ","+Double.toString(maxP)), new Text(idx));
    }


}
