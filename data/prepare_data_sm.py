# Execution of this file leads to preparation of the Netflix dataset into
# the format that is required for the algorithm. 4 new files are created

from pyspark import SparkConf
from pyspark.context import SparkContext
import numpy as np

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
    with open('small_dataset_5.txt', 'w') as f:
        for line in data.toLocalIterator():
            if i == 50000:
                break
            f.write(line + '\n')
            i += 1
            
def reindex_datasets():
    new_index = 0
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    indexes = dict()
    
    for i in range(1,5):
        data = sc.textFile('prepared_data_' + str(i) + '.txt')
        with open('reindexed_data_' + str(i) + '.txt', 'w') as f:
            for line in data.toLocalIterator():
                row = line.split(',')
                if row[1] in indexes:
                    f.write(row[0] + ',' + indexes[row[1]] + ',' + row[2] + '\n')
                else:
                    indexes[row[1]] = str(new_index)
                    f.write(row[0] + ',' + indexes[row[1]] + ',' + row[2] + '\n')
                    new_index += 1
            
        print("Reindexed dataset " + str(i))
        
def train_test_split():
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    
    for i in range(1,5):
        data = sc.textFile('reindexed_data_' + str(i) + '.txt')
        with open('train_data_' + str(i) + '.txt', 'w') as train:
            with open('test_data_' + str(i) + '.txt', 'w') as test:
                for line in data.toLocalIterator():
                    if np.random.randint(100) < 67:
                        train.write(line + '\n')
                    else:
                        test.write(line + '\n')
        
        print("Split dataset " + str(i))


def reindex__small_atasets():
    new_index_item = 0
    new_index_user = 0
    indexes_item = dict()
    indexes_user = dict()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    

    data = sc.textFile('small_dataset_5.txt')
    with open('reindexed_small_dataset_5.txt', 'w') as f:
        for line in data.toLocalIterator():
            row = line.split(',')
            
            if row[0] in indexes_item:
                item = indexes_item[row[0]]
            else:
                indexes_item[row[0]] = str(new_index_item)
                item = indexes_item[row[0]]
                new_index_item += 1
            
            if row[1] in indexes_user:
                user = indexes_user[row[1]]
            else:
                indexes_user[row[1]] = str(new_index_user)
                user = indexes_user[row[1]]
                new_index_user += 1

            f.write(item + ',' + user + ',' + row[2] + '\n')
        
    print("finished reindexing small dataset")
    print("after reindex, there are " + str(new_index_user+1) + "users")
    print("after reindex, there are " + str(new_index_item+1) + "items")

def train_test_split_sm():
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    
    data = sc.textFile('reindexed_small_dataset_5.txt')
    with open('train_small_dataset_5.txt', 'w') as train:
        with open('test_small_dataset_5.txt', 'w') as test:
            for line in data.toLocalIterator():
                if np.random.randint(100) < 67:
                    train.write(line + '\n')
                else:
                    test.write(line + '\n')
        
    print("Split small dataset")

    

if __name__ == '__main__':
    """print("Preparing dataset 1...\n")
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
    print("Finished!")"""
    
    print("Creating a small dataset")
    small_dataset()
    
    print("Reindexing small datasets")
    reindex__small_atasets()
    print("Splitting data into train and test sets")
    train_test_split_sm()
    
    print("Finished!")
