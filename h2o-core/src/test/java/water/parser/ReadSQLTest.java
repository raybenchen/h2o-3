package water.parser;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import water.TestUtil;
import water.fvec.Frame;
import water.jdbc.SQLManager;

import static org.junit.Assert.assertTrue;

public class ReadSQLTest extends TestUtil{
  private String conUrl = "jdbc:mysql://172.16.2.178:3306/ingestSQL?&useSSL=false";
  String user = "root";
  String password = "0xdata";
  boolean optimize = true;
  
  @BeforeClass
  static public void setup() {stall_till_cloudsize(1);}

  @Test
  public void citibike20k() {
    String table = "citibike20k";
    Frame sql_f = SQLManager.readSqlTable(conUrl, table, user, password, optimize).get();
    assertTrue(sql_f.numRows() == 2e4);
    assertTrue(sql_f.numCols() == 15);
    sql_f.delete();
  }
  
  @Test
  public void allSQLTypes() {
    String table = "allSQLTypes";
    Frame sql_f = SQLManager.readSqlTable(conUrl, table, user, password, optimize).get();
    sql_f.delete();
    
  }
  
  @Ignore @Test
  public void airlines() {
    String conUrl = "jdbc:mysql://localhost:3306/menagerie?&useSSL=false";
    String table = "air";
    String password = "ludi";
    Frame sql_f = SQLManager.readSqlTable(conUrl, table, user, password, optimize).get();
    sql_f.delete();
  }
  

}
