import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileUtil;


public class FilePath {
    public static Path[] getFolders(String parentFolder, Configuration conf) throws IOException {
        FileSystem hdfs = FileSystem.get(URI.create(parentFolder), conf);
        FileStatus[] fs = hdfs.listStatus(new Path(parentFolder));
        return FileUtil.stat2Paths(fs);
    }
}
