# import library
import csv
import glob

# class


class MergeFile():
    # inititalization
    def __init__(self, path, newpath):
        self.path = glob.glob(path+"*.csv")
        self.records = []
        self.ID_row_map = {}
        self.newpath = newpath
    # Method to get all data merged in one variable

    def getMergedFile(self):

        i = 0
        for files in self.path:
            if (i == 0):
                with open(files) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # assigning ID of file to variable
                        ID = row["ID"]
                        # assigning entire row as value to ID as key
                        self.ID_row_map[ID] = row
            i = i+1
            if i > 0:
                with open(files) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ID = row["ID"]
                        # Checks if ID is present in ID_row _map or not
                        if ID not in self.ID_row_map:
                            continue
                        # If Id is present it updates the row with same ID
                        self.ID_row_map[ID].update(row)

        # list to get the result in required format.
        for key, value in self.ID_row_map.items():
            self.records.append(value)
        return self.records   
    # Method to write merged data in one file

    def writeintofile(self):
        self.record = self.getMergedFile()
        with open(self.newpath+'op.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, self.record[1].keys())
            dict_writer.writeheader()
            dict_writer.writerows(self.record)
