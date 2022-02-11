# Execution of this file leads to preparation of the Netflix dataset into
# the format that is required for the algorithm. 4 new files are created

from pyspark import SparkConf
from pyspark.context import SparkContext

def prepare_dataset(infile, outfile):
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    data = sc.textFile(infile)
    
    movie = 0
    with open(outfile, 'w') as f:
        for rating in data.toLocalIterator():
            if rating[-1]==':':
                movie = rating[:-1]
                continue
            elements = rating.split(',')
            
            f.write(movie + ',' + elements[0] + ',' + elements[1] + '\n')

def small_dataset():
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    data = sc.textFile('prepared_data_1.txt')
    
    i = 0
    with open('small_dataset.txt', 'w') as f:
        for line in data.toLocalIterator():
            if i == 50000:
                break
            f.write(line + '\n')
            i += 1

if __name__ == '__main__':
    print("Preparing dataset 1...\n")
    prepare_dataset('combined_data_1.txt', 'prepared_data_1.txt')
    print("Finished! \n")
    print("Preparing dataset 2...\n")
    prepare_dataset('combined_data_2.txt', 'prepared_data_2.txt')
    print("Finished! \n")
    print("Preparing dataset 3...")
    prepare_dataset('combined_data_3.txt', 'prepared_data_3.txt')
    print("Finished!")
    print("Preparing dataset 4...")
    prepare_dataset('combined_data_4.txt', 'prepared_data_4.txt')
    print("Finished!")
    print("Creating a small dataset")
    small_dataset()
    print("Finished!")
