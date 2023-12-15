#!/usr/bin/env python
# coding: utf-8
import polars as pl
import numpy as np
import tensorflow as tf

def extract_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts features via regex from raw txt-file input.
    Drops rows which dont contain a truck_id or 3D packing instances
    """
    df = df.with_columns([
        pl.Series(name="index", values=np.arange(len(df.collect()))),
        #pl.Series(name="index", values=np.arange(len(df))),
        pl.col("raw").str.extract("in Truck (\w\d{9})\n").alias("truck_id"),
        pl.col("raw").str.extract("dataset: (\w*)\n").alias("dataset"),
        pl.col("raw").str.extract("instance: ([A-Z0-9]*)\n").alias("instance"),
    ])
    
    return df


def explode_instances_into_stacks(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Use regex to extract all stacks made of item ids, and explode df
    into rows of stacks, belonging to an instance.
    """
    
    df = (
        df.with_columns([
            pl.col("raw").str.extract_all("Stack \d* with items: (\[.*\])\n").alias("stacks")
        ])
          .drop(["raw"])
          .explode(columns = ["stacks"])\
          .with_columns([
              pl.col("stacks").str.extract("Stack (\d*) with").cast(pl.Int64, strict=False).alias("stack_id"),
              pl.col("stacks").str.extract_all("(\d{10}_\d{14})").alias("item_id")
          ])
    )
    return df


def explode_stacks_into_items(df: pl.DataFrame) -> pl.DataFrame:
    """
    Use regex to identify a stack and explode it into items,
    belonging to a stack, which again belongs to an instance.
    """
    df = df.with_columns([
        pl.col("stacks").str.extract("Stack (\d*) with").cast(pl.Int64, strict=False).alias("stack_id"),
        pl.col("stacks").str.extract_all("(\d{10}_\d{14})").alias("item_id")
    ]).drop("stacks").explode("item_id")\
    
    return df


def join_items(df: pl.DataFrame, items) -> pl.DataFrame:
    """
    Perform a simple left join to add item level info.
    
    Validation sadly not supported for multiple keys (yet?)
    """
    
    return df.join(items, how = "left", on = ["dataset", "instance", "item_id"])


def group_items_by_stack(df: pl.DataFrame) -> pl.DataFrame:
    """
    Group item level into back into a stack
    
    - Note that the Width and Length are the same for all items
    - Weight and Height have to be summed up, correcting for nesting height
    - forcedOrientation of a single item makes the whole stack oriented
    - Include Logistics info for stop differentation down the road
    
    """
    
    df = df.group_by(["index", "dataset", "instance", "truck_id", "stack_id"],
                    maintain_order = True)\
    .agg([
        pl.count("item_id").alias("items"),
        pl.max("Length"),
        pl.max("Width"),
        #pl.sum("NestedHeight"),
        #pl.last("Nesting height"),
        pl.sum("Weight"),
        pl.any("ForcedLength"),
        pl.any("ForcedWidth"),
        # Logistic order info
        pl.first("Supplier code"),
        pl.first("Supplier dock"),
        pl.first("Plant dock")
    ]).sort(["index", "stack_id"]).drop(["stack_id"])

    return df


def join_truck_loading_order(df: pl.DataFrame, truck_stops) -> pl.DataFrame:
    """
    Extract the loading orders in the truck_stops
    from the supplier and plant order, given by the stacks/items
    
    TODO:
    -----
    
    - Include a check whether the last item in a stack is different from the first
        This would mean that it is a multistack, making the problem much more difficult
        Jakob said this is often explicitly forbidden by the heuristic, and rarely the 
        case anyways.
    """
    
    truck_join_clms = ["truck_id", "dataset", "instance",
                       "Supplier code", "Supplier dock", "Plant dock"]
    truck_info_clms = ["Supplier loading order",
                       "Supplier dock loading order",
                       "Plant dock loading order"]
    
    df = df.join(truck_stops, how = "left", on = truck_join_clms )
    df = df.drop(["items"])
    df = df.with_columns([
        pl.concat_str(pl.col(truck_info_clms), separator = "-").alias("packing_order")
    ])
    
    df = df.drop(truck_join_clms[-3:]
                 + truck_info_clms
    )
    
    return df
    

def append_truck_info(df: pl.DataFrame, truck_dims: pl.DataFrame) -> pl.DataFrame:
    """
    Add trucks into dataset adding them as rows
    with exactly the same features as the original df
    """

    # Ok, this is a bit messy:
    # since we have to include truck axle load 
    # (in the boolean Forced Columns)
    # We have to cast them to float before the
    # axle load of type float can be merged
    df = df.with_columns([
        pl.col("ForcedWidth").cast(pl.Float64),
        pl.col("ForcedLength").cast(pl.Float64)
    ])



    truck_dims = (
        truck_dims
        .join(df, on = ["dataset", "instance", "truck_id"])
        .unique()
        #.drop(["dataset", "instance", "truck_id"])
    )
    
    df = df.drop(["dataset", "instance", "truck_id"])
    df = df.with_columns([
        pl.lit(False).alias("stack_not_included")
    ])
    
    truck_dims = truck_dims.with_columns([
        pl.col("EMmm").cast(pl.Float64).alias("ForcedLength"),
        pl.col("EMmr").cast(pl.Float64).alias("ForcedWidth"),
        pl.lit("0-0-0").alias("packing_order"),
        pl.lit(False).alias("stack_not_included")
    ])
    
    truck_dims = truck_dims.collect()[df.columns].lazy().unique()

    df = pl.concat([df, truck_dims])
    df = df.sort(["index", "Length"])
    
    return df


def polars_preprocessing_pipeline(
        df: pl.DataFrame,
        items: pl.DataFrame,
        truck_stops: pl.DataFrame,
        truck_dims: pl.DataFrame
    ) -> np.array:
    """
    Takes the raw dataframe with text column and turns it into
    a the numpy version of a df with stacks and trucks as rows,
    which can be grouped by the index column, i.e the first
    column of the numpy array


    """
    
    X = (
        df.lazy()
        .pipe(extract_raw_data)
        .pipe(explode_instances_into_stacks)
        .pipe(explode_stacks_into_items)
        .pipe(join_items, items)
        .pipe(group_items_by_stack)
        .pipe(join_truck_loading_order, truck_stops)
        .pipe(append_truck_info, truck_dims)
        .collect()
        .to_numpy()
    )

    return X




def get_tensor_representation(X, pad_size=80, packing_clm=6):
    """
    Turn the 2D dataframe consisting of example-index,
    stack info and truck info into a 3D Tensor by adding
    an example index dimension and padding the variable length
    stacks to the pad size
    """

    # add columns for Length and Width Remainder
    X = np.append(X, np.zeros((X.shape[0], 2)), axis=1)
    
    indices = np.unique(X[:, 0])
    indices = np.sort(indices)

    # (batch_size, ?, features)
    X = np.array([X[X[:,0] == idx] for idx in indices], dtype = "object")

    # replace the packing order with the stop index (i.e 1-1-1 and 1-1-2 turn to 0 and 1, respectively)
    #packing_clm = min([i for i, clm in enumerate(df.columns) if clm == "packing_order"])
    
    for i, x in enumerate(X):
        packing_order = x[:,packing_clm]
        stops = np.unique(packing_order)
        stops = np.sort(stops)
        stops = {stop: j for j, stop in enumerate(stops)}
        stops = [stops[order] for order in packing_order]
        X[i][:,packing_clm] = stops

    # pad the variable length number of stacks into fixed
    #  (batch_size, pad_size, features)
    X = tf.keras.utils.pad_sequences(X, maxlen=pad_size, padding = "post", dtype="float32")
    # drop the index column (batch_size, pad_len, n_features)
    X = X[:,:,1:].astype("float32")

    # Add Length and width Remainder
    for xx in X:
        truck_width = max(xx[:,1])
        # Length Remainder
        xx[:,-2] = truck_width % xx[:,0]
        # Width Remainder
        xx[:,-1] = truck_width % xx[:,1]
    
    X = np.nan_to_num(X)
    
    return X


def get_labels(df: pl.DataFrame, pad_size=80) -> dict[str: np.array]:
    """
    Take the content of the text-files in the 'raw' column of 
    the polars dataframe and use regex to extract a number of potential labels.

    Note that the labels all stem from after the first MIP 'Improvement',
    which is just the initial solution.
    
    Returns:
    --------

    Y : dict[str: np.array]
        A dictionary containing a list of possible labels
        The length of the arrays equals the length of the df, i.e the batch size

        There are several potential labels that one might want to predict:

        - Solved:
            Has the optimal solution been confirmed?
            
            Note that we do not call it "optimal", because a solution could be optimal
            but the MIP solver did you yet confirm that the solution is indeed optimal

        - Improvement:
            Has an Improvement (other than the initial 'Improvement') been found?
            This is the main goal of our model, the target with the actual impact on the heuristic
            
            There is also the count of improvements, but this makes for a bad target,
            since the order in which solutions are searched determines the number of improvements.
            Eg in one run we find three solutions that are each better than the last.
            If we do the same run again, but happen to seach in the reverse order,
            we only find a single improvement.

        - Area:
            Area and Area Ratio constitute the packed Area of the Truck after optimization.

            This might be useful when the model is able to find an improvement by simply swapping
            a stack with stack that is only a few millimeters wider/longer. This technically this
            constitutes an improvement, it does not affect the bottom line.

            Therefore the Area Forecast can be used as an additional criterion for the extension
            of the time limit (If the predictions are reliably accurate)
            
            
    
    """

    pattern = "Optimal Solution confirmed"
    y_solved = df["raw"].str.contains(pattern)

    pattern = "MIP Improvement - 2D Vol: \d*\.\d* \[m2\] - packed 2D Vol Ratio\: \d*\.\d* \[\%\] - after \d*\.\d* \[s\]"
    mip_improvements = df["raw"].str.extract_all(pattern)#.list[-1][2]

    # mip_improvements: pl.Series[list[str]]
    # with entries according to the pattern, i.e all MIP improvement rows
    
    y_num_improvements = mip_improvements.list.len()-1
    
    y_improvement = y_num_improvements > 0

    y_packed_area_ratio = mip_improvements.list[-1].str.extract("\: (\d*\.\d*) \[\%\]").cast(pl.Float32)

    y_packed_area = mip_improvements.list[-1].str.extract("- 2D Vol: (\d*\.\d*) \[m2\]").cast(pl.Float32)

    y_first_update = mip_improvements.list[1].str.extract("- after (\d*\.\d*) \[s\]").cast(pl.Float32).fill_null(0)
    
    y_last_update = mip_improvements.list[-1].str.extract("- after (\d*\.\d*) \[s\]").cast(pl.Float32)

    # missing stacks:
    y_stack_not_included = np.zeros((len(df), pad_size), dtype=float)
    pattern = "Stack (\d*) not in final solution with items:"
    x = df["raw"].str.extract_all(pattern).map_elements(lambda x: [int(i.split(" ")[1]) for i in x])
    
    for i, missing_stacks in enumerate(x):
        for j in missing_stacks:
            y_stack_not_included[i, j] +=1

    Y = {
        "Solved": y_solved.to_numpy().astype("float32"),
        "Improvement": y_improvement.to_numpy().astype("float32"),
        "Number_Improvements": y_num_improvements.to_numpy().astype("float32"),
        "AreaRatio": y_packed_area_ratio.to_numpy().astype("float32"),
        "Area": y_packed_area.to_numpy().astype("float32"),
        #np.log1p(y_first_update.to_numpy()),
        #y_last_update.to_numpy(),
        "Stacks": y_stack_not_included.astype("float32"),
    }
    
    return Y



def input_output_generation(
        X_batch,
        items: pl.DataFrame,
        truck_stops: pl.DataFrame,
        truck_dims: pl.DataFrame,
        target_labels=["Solved", "Improvement", "AreaRatio", "Stacks"],
        shuffle=True,
        pad_size=80,
    ) -> np.array:
    """

    Returns:
    --------
    X: np.array[float32]
        3D Feature Tensor of shape (Batch_size, Pad_size, n_features=7)

        - Batch_size: Truck Optimization Instances
        - Pad_size: Stacks (or Trucks), padded up to create tensors
        - n_features: Length, Width, Weight, L/W Forced Orientation
                      packing order, is_truck
    """
    
    df = pl.DataFrame({"raw": X_batch.numpy().astype(str)})
    X = polars_preprocessing_pipeline(
            df,
            items=items,
            truck_stops=truck_stops,
            truck_dims=truck_dims,
        )
    X = get_tensor_representation(X, pad_size=pad_size)

    # fill final column with bool for stack not in initial solution
    pattern = "Stack (\d*) missing:"
    x = df["raw"].str.extract_all(pattern).map_elements(lambda x: [int(i.split(" ")[1]) for i in x])

    for i, missing_stacks in enumerate(x):
        for j in missing_stacks:
            X[i, j, 6] +=1

    
    # extract the time limit
    pattern = "2D Packing MIP with Time Limit (\d*\.?\d*) \[s\]"
    x_time_limit = df["raw"].str.extract(pattern).cast(pl.Float32).to_numpy()

    Y = get_labels(df, pad_size=pad_size)
    Y = [Y[target] for target in target_labels]

    if shuffle:
        idx = np.arange(X.shape[1])
        idx = np.random.choice(idx, size=pad_size, replace=False)
        X = X[:,idx,:]

        # if you shuffle the input order of stacks
        # you also have to shuffle the output order of stacks
        if "Stacks" in target_labels:
            stack_idx = ("Stacks" == np.array(target_labels))
            stack_idx = np.where(stack_idx)
            stack_idx = stack_idx[0][0]
            Y[stack_idx] = Y[stack_idx][:,idx]

    X = [X, x_time_limit]
    
    return X, Y

