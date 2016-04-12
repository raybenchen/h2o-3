package water.api;


import water.Job;
import water.jdbc.SQLManager;

/**
 * Import Sql Table into H2OFrame
 */

public class ReadSQLTableHandler extends Handler {

  @SuppressWarnings("unused") // called through reflection by RequestServer
  public JobV3 readSQLTable(int version, ReadSQLTableV99 readSQLTable) {
     Job j = SQLManager.readSqlTable(readSQLTable.connection_url, readSQLTable.table, readSQLTable.username,
             readSQLTable.password, readSQLTable.optimize);
    return new JobV3().fillFromImpl(j);
    
  }

}
