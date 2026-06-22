import os
import sys
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
sys.path.append(str(path_root))
# ______________________________________________________

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from preprocessing.routines import *

SOURCE_PATH              = "/test/source"
# Nacho
SINK_PATH                = "/test/sink"
SOURCE_DATASET_PATH      = "{}/datasets".format(SOURCE_PATH)
SINK_DATASET_PATH        = SINK_PATH
DATASET_DESCRIPTION_PATH = "{}/datasets_desc3.csv".format(SOURCE_PATH)


def generate_dataset_description_report(**kwargs):
    filename             = kwargs.get("filename")
    extension            = kwargs.get("extension","txt") 
    dataset_descriptions = kwargs.get("dataset_descriptions")
    write                = kwargs.get("write",False)
    full_path            = "{}/{}.{}".format(SINK_DATASET_PATH,filename,extension)
    f                    = open(full_path,"w") if(write)  else None
    #with open(full_path,"w") as f:
    for index,(key,value) in enumerate(dataset_descriptions.items()):
        separator = value.separator
        le = preprocessing.LabelEncoder()
        if(separator):
            df_path          = "{}/{}".format(SOURCE_DATASET_PATH,value.fullname)
            df               = pd.read_csv(df_path,sep = separator, header= None if not (value.remove_headers) else 'infer')
            df               = remove_headers(df,value.remove_headers)
            df,vector_class  = remove_vector_class(df,value.vector_class_index)
            raw_vector_class = vector_class.values.flatten()[1:]
            le.fit(raw_vector_class)
            target           = le.transform(raw_vector_class)
            vector_class     = pd.DataFrame(
                {
                    "target":target
                }
            )
            df              = df.apply(pd.to_numeric,errors="coerce").fillna(0)
            df              = remove_string_columns(df)
            x               = vector_class.value_counts()
            xx              = pd.DataFrame( {"CLASS":list(range(x.size)),"COUNT":x.values } )
            if(write):
                f.write(key+"\n")
                f.write("&"+str(df.shape[0])+"\n")
                f.write("&"+str(df.shape[1])+"\n")
                f.write("&"+str(len(x))+"\n")
                f.write("\n")
            df.to_csv("{}/{}.{}".format(SINK_DATASET_PATH,value.filename,"csv"),index=False,header=None)
            xx.to_csv("{}/{}.csv".format(SINK_DATASET_PATH,value.filename+"_counter","csv"),index=False)
            vector_class.to_csv("{}/{}.{}".format(SINK_DATASET_PATH,value.filename+"_target","csv"),index=False)
            print("PROCESSED_SUCCESSFULLY",value)
            print("_"*30)
    if(write):
        f.close()

def main():
    
    fullnames               = os.listdir(SOURCE_DATASET_PATH)  # List of all the filenames in BASE_PATH.
    print(fullnames)
    fullnames.sort() # Sort the filenames alphabetically.
    datasets_desc_df        = pd.read_csv(DATASET_DESCRIPTION_PATH) # Dictionary save all the dataset descriptions
    dataset_descriptions    = init_data_descriptions_from_df(
        df = datasets_desc_df,
        fullnames = fullnames
    )
    generate_dataset_description_report(
        filename             = "report",
        extension            = "txt",
        dataset_descriptions = dataset_descriptions,
        write                = False
    )


if __name__ == '__main__':
	main()