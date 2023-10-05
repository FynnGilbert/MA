#!/usr/bin/env python
# coding: utf-8

import os
import polars as pl
from tqdm import tqdm


cwd = os.getcwd()
cwd, _ = os.path.split(cwd)
folder = "datasets"
root = os.path.join(cwd, folder)


item_files = []
truck_files = []

for dataset in os.listdir(root):
    print(dataset)
    dataset_path = os.path.join(root, dataset)
    for instance in os.listdir(dataset_path):
        instance_path = os.path.join(dataset_path, instance)
        for input_file in os.listdir(instance_path):
            print(dataset, instance, input_file, end= "\t")
            data_path = os.path.join(instance_path, input_file)
            if "input_items.csv" == input_file:
                print("INCLUDED")
                item_files.append((dataset, instance, input_file))
            elif "input_trucks.csv" == input_file:
                print("INCLUDED")
                truck_files.append((dataset, instance, input_file))
            else:
                print("NOT")


for i, item_file in enumerate(tqdm(item_files)):
    dataset, instance, file = item_file
    path = os.path.join(root, *item_file)
    
    data = pl.read_csv(path, separator = ";")
    data = data.with_columns([
        pl.lit(dataset[8:]).alias("dataset"),
        pl.lit(instance).alias("instance"),
        pl.col("Weight").str.replace(",", ".").cast(float),
    ]).unique()
    
    if i == 0:
        items = pl.DataFrame(schema=data.schema)
    
    items = items.extend(data)
    

for i, item_file in enumerate(tqdm(truck_files)):
    dataset, instance, file = item_file
    path = os.path.join(root, *item_file)
    
    data = pl.read_csv(path, separator = ";")
    data = data[:,:28]
    data = data.with_columns([
        pl.lit(dataset[8:]).alias("dataset"),
        pl.lit(instance).alias("instance"),
        pl.col("EM").str.replace(",", ".").cast(float),
        pl.col("CM").str.replace(",", ".").cast(float),
    ]).unique()
    
    if i == 0:
        trucks = pl.DataFrame(schema=data.schema)
        clms = trucks.columns
    
    trucks = trucks.extend(data)



if __name__ == "__main__":
    truck_path = os.path.join(cwd, "truck-item-infos", "trucks.csv")
    trucks.write_csv(truck_path)

    item_path = os.path.join(cwd, "truck-item-infos", "items.csv")
    items.write_csv(item_path)

