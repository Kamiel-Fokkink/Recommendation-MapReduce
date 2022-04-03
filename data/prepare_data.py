# Execution of this file leads to preparation of the Netflix dataset into
# the format that is required for the algorithm. 4 new files are created

from pyspark import SparkConf
from pyspark.context import SparkContext
import numpy as np

def prepare_dataset(infile, outfile):
    """Transform dataset from format: movie: [list of (user, rating time)]
    into a usable format for us: (movie, user, rating)"""
    
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    data = sc.textFile(infile)
    
    movie = 0
    with open(outfile, 'w') as f:
        for rating in data.toLocalIterator():
            # Check if line corresponds to a new movie
            if rating[-1]==':':
                movie = rating[:-1]
                continue
            
            # For a rating, write it down in correct format
            elements = rating.split(',')
            f.write(movie + ',' + elements[0] + ',' + elements[1] + '\n')

def small_dataset():
    """Make a small dataset of size 50000 just for purposes of testing locally"""
    
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    data = sc.textFile('ptrain_data_1.txt')
    
    i = 0
    with open('small_dataset.txt', 'w') as f:
        for line in data.toLocalIterator():
            if i == 50000:
                break
            f.write(line + '\n')
            i += 1
            
def reindex_datasets():
    """Reindex the user_ids in the test set. As these are first random large
    unconsecutive numbers, our matrix H gets way bigger than it needs to be.
    Reindex into a new range from 0 to n_users."""
    
    new_index = 0
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    indexes = dict()
    
    # Loop over all datafiles
    for i in range(1,5):
        data = sc.textFile('prepared_data_' + str(i) + '.txt')
        with open('reindexed_data_' + str(i) + '.txt', 'w') as f:
            for line in data.toLocalIterator():
                row = line.split(',')
                
                # If user has been seen before, assign to that user
                if row[1] in indexes:
                    f.write(row[0] + ',' + indexes[row[1]] + ',' + row[2] + '\n')
                
                # If user is new, allocate the next consecutive user_id
                else:
                    indexes[row[1]] = str(new_index)
                    f.write(row[0] + ',' + indexes[row[1]] + ',' + row[2] + '\n')
                    new_index += 1
            
        print("Reindexed dataset " + str(i))
        
def train_test_split():
    """Make a train-test split on the data. Randomly allocate datapoints into
    one or the other, based on a 33/100 ratio."""
    
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    
    # Loop over all data files
    for i in range(1,5):
        data = sc.textFile('reindexed_data_' + str(i) + '.txt')
        with open('train_data_' + str(i) + '.txt', 'w') as train:
            with open('test_data_' + str(i) + '.txt', 'w') as test:
               
                # Allocate each line into train or test set
                for line in data.toLocalIterator():
                    if np.random.randint(100) < 67:
                        train.write(line + '\n')
                    else:
                        test.write(line + '\n')
        
        print("Split dataset " + str(i))
            
    

if __name__ == '__main__':
    """Run this to perform all steps of preprocessing, reindexing and 
    train-test split one after the other. May take 10-20 minutes."""
    
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
    
    print("Reindexing datasets")
    reindex_datasets()
    print("Splitting data into train and test sets")
    train_test_split()
    #print("Creating a small dataset")
    #small_dataset()
    print("Finished!")
