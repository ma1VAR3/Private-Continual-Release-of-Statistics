import numpy as np
import pandas as pd
import h3

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

# separating latitude and longitude from location
lat_lon = df["location.coordinates"].astype(str).str.strip('[]').str.split(",")
lon = lat_lon.apply(lambda x: x[0])
lat = lat_lon.apply(lambda x: x[1])

# assigning h3 index to the latitude and longitude coordinates in separate dataframe 
dflen = len(df)
h3index = [None] * dflen
resolution = 7
for i in range(dflen):
    h3index[i] = h3.geo_to_h3(lat=float(lat[i]), lng=float(lon[i]), resolution=resolution)
    
df["h3index"] = h3index

# assigning date and time to separate dataframe and creating a timeslot column
df["Date"] = pd.to_datetime(df["observationDateTime"]).dt.date
df["Time"] = pd.to_datetime(df["observationDateTime"]).dt.time
time = df["Time"]
df["Timeslot"] = time.apply(lambda x: x.hour)

# assigning HATs from H3index and timeslot
df["HAT"] = (df["Timeslot"].astype(str) + " " + df["h3index"])

# Filtering time slots by start and end time 
startTime = 9
endTime = 20
df = df[(df["Timeslot"] >= startTime) & (df["Timeslot"] <= endTime)]

# Selecting h3 indices where a min number of events occur in all timeslots of the day
tmp_df1 = (df.groupby(["Timeslot", "Date", "h3index"]).agg({
    "license_plate": "nunique"
}).reset_index())

tmp_df2 = tmp_df1.groupby(["Timeslot", "h3index"]).agg({
    "license_plate": "sum"
}).reset_index()

date = df["Date"].unique()
minEventOccurences = 20
limit = len(date) * minEventOccurences

tmp_df3 = tmp_df2[tmp_df2["license_plate"] >= limit]
tmp_df4 = tmp_df3.groupby("h3index").agg({"Timeslot": "count"}).reset_index()
maxTimeSlots = tmp_df4["Timeslot"].max()
tmp_df5 = tmp_df4[tmp_df4["Timeslot"] == maxTimeSlots]

t_df = df["h3index"].isin(tmp_df5["h3index"])
# print(t_df.head())
df = df[t_df]

print(df.head(10))
