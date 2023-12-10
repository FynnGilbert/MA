#!/usr/bin/env python3

import os
import re
import polars as pl
from tqdm import tqdm
import hashlib



def read_file(path: str) -> str:
    
    assert path.endswith(".out") # path is a txt
    assert os.path.isfile(path)  # file exists
    
    with open(path, "r") as file:
        text = file.read()
    return text


def md5hash(s: str): 
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def read_path_to_df(path: str) -> pl.DataFrame | pl.LazyFrame:
    """
    Get a filepath, read its data
    """
    data = read_file(path)
    data = data.split("-"*100)
    data = data[1:]
    data = pl.DataFrame(data, schema = {"raw": str})
    data = data.filter(
        pl.col("raw").str.len_bytes() > 3
    )[:-1]
    
    path, instance = os.path.split(path)
    instance = instance[:-4]
    path, dataset = os.path.split(path)
    
    #_, dataset, instance = re.split("\s{1,}", data[0, 0].split("\n")[1])
    #data[0, 0] = "\n".join(data[0, 0].split("\n")[3:])

    data = data.with_columns([
        # strip leading and trailing whitespace
        pl.col("raw").str.strip_chars_start().str.strip_chars_end(),
        pl.lit(dataset).alias("dataset"),
        pl.lit(instance).alias("instance"),
        pl.col("raw").str.extract("(\d)D Packing ").alias("D"),
        pl.col("raw").str.extract("\dD Packing (\w*)\s").alias("model"),
        pl.col("raw").str.contains("Solve interrupted after").alias("interrupted"),
        pl.col("raw").map_elements(md5hash).alias("hash")
    ])

    data = data.drop_nulls()
    
    return data

test_instances = [
 'BY',
 'BY2',
 'BY3',
 'FS',
 'FS2',
 'FS3',
 'MA4',
 'MA5',
 'MA6',
 'MA7',
 'PA',
 'PA2',
 'PA3',
 'SA4',
 'SA5',
 'SA6',
 'SA7',
 'TA',
 'TA2',
 'TA3',
 'TR4',
 'TR5',
 'TR6',
 'TR7'
]

hold_instances = [
 'BY4',
 'BY5',
 'BY6',
 'BY7',
 'CC4',
 'CC5',
 'CC6',
 'CC7',
 'DO4',
 'DO5',
 'DO6',
 'DO7',
 'FS4',
 'FS5',
 'FS6',
 'FS7',
 'PA4',
 'PA5',
 'PA6',
 'PA7',
 'TA4',
 'TA5',
 'TA6',
 'TA7'
]


def df_to_labeled_instance_files(df: pl.DataFrame, cwd:str) -> None:
    """
    Creates a file for each row in the dataframe in the corresponding folder
    """
    root = os.path.join(cwd, "data")
    
    for row in df.iter_rows():
        raw, dataset,instance, D, model, interrupted, hash_ = row
        
        text = f"dataset: {dataset}\ninstance: {instance}\n+" + "-"*11 +"+\n"
        text += raw
        #print(text)
        
        if dataset.startswith("X"):
            if instance in hold_instances:
                path = os.path.join(root, "holdout")
            elif instance in test_instances:
                path = os.path.join(root, "test")
            else:
                path = os.path.join(root, "train")
        else:
            path = os.path.join(root)

        if D == "3":
            #path = os.path.join(root, "3D")
            continue
        elif D == "2":
            path = os.path.join(path, "2D")
            
            if (model == "Heuristik") | (model == "Heuristic"):
                #path = os.path.join(path, "Heuristic")
                continue
            elif model == "MIP":
                path = os.path.join(path, "MIP")
                
                if interrupted:
                    path = os.path.join(path, "interrupted")
                else:
                    path = os.path.join(path, "solved")
                
            else:
                print("Model is neither MIP nor Heuristic:")
                for clm in row:
                    print(clm, sep = "\n\n")
                #raise AssertionError
                continue
    
            
        else:
            print("Model D is neither 2 nor 3:")
            print(row)
            #raise AssertionError
            continue
        
        path = os.path.join(path, hash_+".txt")
        if os.path.exists(path):
            continue
        with open(path, "w") as file:
            file.write(text)

    return None






if __name__ == "__main__":
    cwd = os.getcwd()
    cwd, _ = os.path.split(cwd)
    ROOT = os.path.join(cwd, "raw-data")

    all_paths = []
    for root, dirs, files in tqdm(os.walk(ROOT, topdown=False)):
        for name in files:
            if not name.endswith(".out"):
                continue
            path = os.path.join(root, name)
            all_paths.append(path)
    
    for path in tqdm(all_paths):
        #print(path, end="\r")
        df = read_path_to_df(path)
        df_to_labeled_instance_files(df, cwd)


