import json
import math
import numpy as np
import pandas as pd
import h3

def load_data(dataset="ITMS", config=None):
    data = None
        
    if dataset == "ITMS":
        df =pd.read_csv("./suratITMSDPtest/suratITMSDPtest.csv")
        df = df.drop_duplicates(subset=['trip_id', 'observationDateTime'], ignore_index=True)
        df = df.drop(columns = [
                    "trip_direction",
                    "last_stop_id",
                    "last_stop_arrival_time",
                    "route_id",
                    "actual_trip_start_time",
                    "trip_delay",
                    "vehicle_label",
                    "id",
                    "location.type",
                    "trip_id"
                ])
        lat_lon = df["location.coordinates"].astype(str).str.strip('[]').str.split(",")
        lon = lat_lon.apply(lambda x: x[0])
        lat = lat_lon.apply(lambda x: x[1])
        dflen = len(df)
        h3index = [None] * dflen
        resolution = config["H3 Resolution"]
        for i in range(dflen):
            h3index[i] = h3.geo_to_h3(lat=float(lat[i]), lng=float(lon[i]), resolution=resolution)
        df["h3index"] = h3index
        df["Date"] = pd.to_datetime(df["observationDateTime"]).dt.date
        df["Time"] = pd.to_datetime(df["observationDateTime"]).dt.time
        time = df["Time"]
        df["Timeslot"] = time.apply(lambda x: x.hour)
        df["HAT"] = (df["Timeslot"].astype(str) + " " + df["h3index"])
        startTime = config["Start time"]
        endTime = config["End time"]
        df = df[(df["Timeslot"] >= startTime) & (df["Timeslot"] <= endTime)]
        df = df[df["speed"]>0]
        df_gb_h_al = df.groupby(["HAT"]).agg({"license_plate": "nunique"}).reset_index()
        max_hat = df_gb_h_al[df_gb_h_al["license_plate"] == df_gb_h_al["license_plate"].max() ]["HAT"].iloc[0]
        h_d = df[df["HAT"]==max_hat]
        h_d = h_d.drop(columns = [
            "observationDateTime",
            "location.coordinates",
            "h3index",
            "Date",
            "Time",
            "Timeslot",
            "HAT"
        ])
        h_d = h_d.rename(columns = {
            'license_plate':'User',
            'speed' : 'Value'
        })
        data = h_d
    return data

def calc_user_array_length(data, type="median"):
    L = None
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    if type=="median":
        L = np.median(data_grouped["Value"])
    elif type=="mean":
        L = math.floor(np.mean(data_grouped["Value"]))
    elif type=="max":
        L = np.max(data_grouped["Value"])
    elif type=="rms":
        L = math.floor(math.sqrt(np.mean([math.pow(i, 2) for i in data_grouped["Value"]])))
    
    return L