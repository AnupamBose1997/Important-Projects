# import libraries
import csv
import glob
import unittest


def getMergedFile(path):
    # Import all the csv files in the folder
    cfiles = glob.glob(path+"*.csv")
    # new dictionary to update all the data
    ID_row_map = {}
    i = 0
    for files in cfiles:
        if (i == 0):
            with open(files) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    #assigning ID of file to variable
                    ID = row["ID"]
                    # assigning entire row as value to ID as key
                    ID_row_map[ID] = row
        i = i+1
        if i>0:
            with open(files) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ID = row["ID"]
                    #Checks if ID is present in ID_row _map or not
                    if ID not in ID_row_map:
                        continue
                    # If Id is present it updates the row with same ID
                    ID_row_map[ID].update(row)

    #Initiate a list to get the result in required format.                
    records=[]
    for key,value in ID_row_map.items():
        records.append(value)

    # write the data in output file.
    with open(path+'op.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, records[1].keys())
        dict_writer.writeheader()
        dict_writer.writerows(records)


# Function Call
getMergedFile("C:/Users/ASUS/Documents/visual studio/")




        
