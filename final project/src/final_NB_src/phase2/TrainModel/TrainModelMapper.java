package TrainModel;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class TrainModelMapper extends Mapper<Object, Text, Text, Text> {
    //  使用多项式分布的朴素贝叶斯来进行文本分类，需要统计的内容有：训练集所有的总词数（包含重复），
    //训练集的词典条目数， 各个类别的的单词总数（含重复）， 某个类别下某个单词出现的次数（含重复）
    //训练集的总词数存放在CountVector/Vector的part-r-00000中
    @Override
    public void map(Object key, Text value, Context context)
    throws IOException, InterruptedException {
        String[] tr = value.toString().split("\t");//分割得到类别和属性列表
        //因为需要统计总词数，所以还需要计算该样本中的总词数
        //  发射的内容有两部分，一个是该样本的类别及其中的单词数<样本类别，单词数>,
        //另一个是该样例中某个单词及其出现的频率<样本类别#单词序号，出现次数>
        String cls = tr[0];
        String[] attributes = tr[1].split(" ");
        long oneSampleWordCnt = 0;
        for(String attr: attributes){
            String wordNo = attr.split(":")[0];
            String wordCnt = attr.split(":")[1];
            oneSampleWordCnt += Long.parseLong(wordCnt);
            context.write(new Text(cls + "#" + wordNo), new Text(wordCnt));
        }
        context.write(new Text(cls), new Text(Long.toString(oneSampleWordCnt)));
    }
}
