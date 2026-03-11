import argparse
import yaml
from datetime import datetime
import toolBox.database as database

def main(config_path):
    start = datetime.now()
    roof_db = database.modelDatabase(config_path)
    roof_db.creeateDB()
    roof_db.createTables()
    roof_db.fillTablesFromDXF()
   
    end = datetime.now()
    print("CAD to Database [Beam, Joint]:\t", (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Roof Model Transfer: DXF to Database")
    parser.add_argument('confFile', type=argparse.FileType('r'), help='Config file (.yml)')
    args = parser.parse_args()

    config_path = args.confFile.name
    main(config_path)