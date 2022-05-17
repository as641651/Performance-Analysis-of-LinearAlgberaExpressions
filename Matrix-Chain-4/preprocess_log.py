import csv
import sys
import math

def decimal_min_to_hh_mm_ss(decimal_min):
        frac,hr = math.modf(float(decimal_min)/60.)
        frac,min = math.modf(frac*60.)
        sec = int(frac*60)
        return "{:02d}:{:02d}:{:02d}".format(int(hr),int(min),sec)
        


if __name__ == "__main__":

    file_path = sys.argv[1]
    new_file = file_path.split(".")[0] + "_processed." + file_path.split(".")[1]

    f1 = open(file_path, 'r',encoding='UTF8')
    f2 = open(new_file, 'w',encoding='UTF8')
    
    csv_reader = csv.reader(f1)
    csv_writer = csv.writer(f2)

    header = next(csv_reader)
    csv_writer.writerow(header)

    for row in csv_reader:
        row[2] = decimal_min_to_hh_mm_ss(row[2])
        row[3] = decimal_min_to_hh_mm_ss(row[3])
        csv_writer.writerow(row)
    
    f1.close()
    f2.close()




