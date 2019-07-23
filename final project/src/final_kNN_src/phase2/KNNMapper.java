import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

public class KNNMapper  extends Mapper<Object, Text, Text, Text> {
    //读入训练集
    //private ArrayList<ArrayList<String>> trainVectors = new ArrayList<ArrayList<String>>();
    //setup中，取得超参数，将文件读取并储存在内存中，文件内容用字符串数组储存，字符串数组首为类别，之后储存训练集文档向量
    //一个变量存储训练集的属性，一个变量存储训练集的类别
    //其中属性因为很稀疏，就不采用arraylist而采用hashmap
    private ArrayList<HashMap<Integer, Double>> trainX = null;
    private ArrayList<String> trainY = null;
    private int kNeighbor = 0;
    private double allT = 0;
    private double allF = 0;


    private double diffFromOne(Double number){
        return abs(number - 1);
    }

    public void setup(Context context)
            throws IOException, InterruptedException{
        //String fileName = context.getProfileParams();//取得超参数
        //        File f = new File(fileName);
        //        Path TraingPath = null;
        //            //TraingPath = new Path(fileName+fileList[0]);

        if(trainX == null){
            trainX = new ArrayList<>();
        }
        else{
            System.err.print("wrongful train X");
            System.exit(-1);
        }
        if(trainY == null){
            trainY = new ArrayList<>();
        }
        else{
            System.err.print("wrongful train Y");
            System.exit(-1);
        }
        URI[] cacheFile = context.getCacheFiles();
        FileSystem fs = FileSystem.get(context.getConfiguration());
        Path trainVecPath = new Path(cacheFile[0]);
        FileStatus[] fss = fs.listStatus(trainVecPath);
        Path[] fullVec = FileUtil.stat2Paths(fss);
        for(Path p: fullVec){
            FSDataInputStream inStream = fs.open(p);
            while(inStream.available() > 0){
                String oneSample = inStream.readLine();
                String[] oneSampleSplit = oneSample.split("\t");
                trainY.add(oneSampleSplit[0]);
                //拆分 训练集样本的X部分（属性部分）
                String[] strAttributes = oneSampleSplit[1].split(" ");
                HashMap<Integer, Double> attributes = new HashMap<>();
                for(String attr: strAttributes){
                    String number = attr.split(":")[0];
                    String val = attr.split(":")[1];
                    String fixVal = val.replace(",", "");
                    attributes.put(Integer.parseInt(number), Double.parseDouble(fixVal));
                }
                trainX.add(attributes);
            }
        }
        kNeighbor = Integer.parseInt(context.getProfileParams());
        /*for(String str:fileList){
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

        }*/
    }


    //所有测试样本都放在一个文件中。目前K设为10
    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        FileSplit fileSplit = (FileSplit) context.getInputSplit();
        String fileName = fileSplit.getPath().getName();//得到文件名

        if (fileName.contains("_SUCC")) {
            return;
        } else {
            String[] testVector = value.toString().split("\t");//注意，字符串数组索引0的位置是文件名
            String sampleName = testVector[0];
            String[] strAttributes = testVector[1].split(" ");
            HashMap<Integer, Double> oneSample = new HashMap<>();
            for(String attr: strAttributes){
                String num = attr.split(":")[0];
                String val = attr.split(":")[1];
                String fixVal = val.replace(",", "");
                oneSample.put(Integer.parseInt(num), Double.parseDouble(fixVal));
            }
            double[][] KNN = new double[kNeighbor][2];//储存最近邻，每行储存距离和该点在训练集中的索引
            for (double[] vector : KNN) {
                vector[0] = Double.POSITIVE_INFINITY;
                vector[1] = -1;
            }
            // new :计算余弦相似度, 越接近1越相似！
            // 因为采用了稀疏矩阵，所以对两个样本的属性都要遍历一次才能计算余弦相似度
            for (int index = 0; index < trainX.size(); index++) {
                double distance = 0;
                double xSum = 0;
                double ySum = 0;
                for(Integer attr: trainX.get(index).keySet()){
                    if(oneSample.keySet().contains(attr)){
                        distance += oneSample.get(attr) * trainX.get(index).get(attr);
                        xSum += trainX.get(index).get(attr) * trainX.get(index).get(attr);
                        ySum += oneSample.get(attr) * oneSample.get(attr);
                    }
                    else{
                        xSum += trainX.get(index).get(attr) * trainX.get(index).get(attr);
                    }
                }
                for(Integer attr: oneSample.keySet()){
                    if(!trainX.get(index).keySet().contains(attr)){
                        ySum += oneSample.get(attr)*oneSample.get(attr);
                    }
                }
                if(xSum == 0 || ySum == 0)//可能出现的0向量
                    distance = -1;
                else{
                    distance = distance /(sqrt(xSum) * sqrt(ySum));
                }
                //old: 计算余弦相似度
                //for (int i = 1; i < trainVectors.get(index).size(); i++)
                //    distance += Double.parseDouble((String) trainVectors.get(index).get(i)) * Double.parseDouble(testVector[i]);
                int replaceIndex = -1;
                for (int i = 0; i < kNeighbor; i++) {
                    if (diffFromOne(KNN[i][0]) > diffFromOne(distance)) {
                        //如果刚开始寻找，或者新找到的点还要原先寻找到的点还要远
                        if ((replaceIndex == -1) || (diffFromOne(KNN[i][0]) > diffFromOne(KNN[replaceIndex][0]))) {
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

            for (int i = 0; i < kNeighbor; i++) {
                if (className.contains(trainY.get((int)KNN[i][1]))) {
                    int index = className.indexOf(trainY.get((int)KNN[i][1]));
                    classCount.set(index, classCount.get(index) + 1);
                } else {
                    className.add(trainY.get((int)KNN[i][1]));
                    classCount.add(1);
                }
            }


            //寻找类别最多的

            int index = -1;

            for (int i = 0; i < classCount.size(); i++) {
                if ((index == -1) || (classCount.get(i) > classCount.get(index))) {
                    index = i;
                }
            }

            context.write(new Text(sampleName), new Text(className.get(index)));
        }
    }

}
