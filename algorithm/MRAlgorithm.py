# Make necessary imports
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.mllib.random import RandomRDDs
from pyspark.sql import SQLContext
from pyspark import sql
import numpy as np



'''
normalization for the initialization of H
'''
def normalization(x):
    s = sum(x[1])
    s_last = 0
    x_new = []
    for i in range(len(x[1])-1):
        x_new.append(x[1][i] / s)
        s_last += x_new[i]
    x_new.append(1 - s_last)
    return (x[0], np.array(x_new))

def create_uniform_rdd_rowindexed_matrix(sc, nrow, ncol, normalize=False):
    rdd = []
    for i in range(nrow):
        rdd.append((i, np.random.rand(ncol)))
    #print(rdd)
    rdd = sc.parallelize(rdd)
    if normalize:
        rdd = rdd.map(normalization)
    return rdd

'''
    updating H function
    used in compute_H
'''
def update_h(hxy_tuple):
    hx, y = hxy_tuple
    h, x = hx
    d = h @ y # a scalar(D_ii)
    e = h @ x.reshape(len(x[0]))
    h_new = h * np.divide(x+d, y+e) # broadcase add
    return h_new

'''
calculating the probability of each user item pair
used in RM2_distribution
'''
def compute_p_i_R(ratings, sum_ratings, item_probs, ld):
    sum_ratings = dict(sum_ratings)
    item_probs = dict(item_probs)
    user_items = dict()
    
    for user, item, rating in ratings:
        total_ratings = sum_ratings[user]
        total_items = item_probs[item]
        prob = ((1 - ld) * rating / total_ratings) + ld * total_items
            
        if user in user_items:
            user_items[user][item] = prob
        else:
            user_items[user] = dict()
            user_items[user][item] = prob
    
    user_item_pairs = set()
    for user in user_items.keys():
        for item in item_probs.keys():
            user_item_pairs.add((user, item))
    
    out = dict()
    for user, item in user_item_pairs:
        
        prod = 1
        for item_t in user_items[user].keys():
            
            sm = 0
            for user_j in user_items.keys():
                p_item_j = user_items[user_j].get(item, 0)
                p_t_j = user_items[user_j].get(item_t, 0)
                sm += p_item_j * p_t_j
                
        prod *= sm
        
        out[(item, user)] = prod
    
    return out

'''
The main algorithm class
'''
class MRAlgorithm:
    
    def __init__(self, datapath, output_path, k, ld):
        self.output_path = output_path
        self.k = k
        self.ld = ld
        #self.n = 2649429
        self.n = 480189
        self.m = 17770
        self.sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.driver.memory", "15g"))
        self.input = self.sc.textFile(datapath).map(lambda x: (int(x.split(',')[0])-1,
                                                               int(x.split(',')[1]),int(x.split(',')[2])))
        self.A = self.input
        self.H = create_uniform_rdd_rowindexed_matrix(self.sc, self.n, self.k, normalize=True)
        self.W = create_uniform_rdd_rowindexed_matrix(self.sc, self.m, self.k)
        
    def compute_H(self):
        print("Compute H")
    
        ## compute X
        # M1
        W_b = self.sc.broadcast(self.W.collect()) # a dictionary where the key is the index i
        Aw = self.A.map(lambda x: (x[1], x[2]*W_b.value[x[0]][1])) #Wi.T
        #print(Aw.collect())
        # R1 
        self.X = Aw.reduceByKey(lambda x, y: x+y) # add a combiner? #.map(lambda x: (x[0], x[1]))
    
        
        ## compute B
        # M2
        ww = self.W.map(lambda x: (None, np.outer(x[1],x[1]))) # w.T.dot(w)? #import numpy.linalg
        # from pyspark.mllib.linalg.distributed import *
        # as_block_matrix(w.T).multiply(as_block_matrix(w))
        # https://stackoverflow.com/questions/37766213/spark-matrix-multiplication-with-python
        # R2
        B = ww.reduceByKey(lambda x, y: x+y).map(lambda x: x[1]) # without map?
        
        ## compute Y
        # M3
        B_b = self.sc.broadcast(B.collect())
        y = self.H.map(lambda x: (x[0], B_b.value@x[1]))
        
        ## update H
        # M4
        self.hxj = self.H.join(y).join(self.X)
        # R4
        # why is it a reduce phase instead of a map phase?
        #hnew = self.hxj.map(lambda x: (x[0], update_h(x[1])))
        #self.H = hnew
        
    def compute_W(self):
        print("Compute W")
        #M5: Compute matrix S (repartition join of H and A)
        tmp_M5 = self.H.join(self.A.map(lambda x: (x[1], (x[0], x[2]))))

        #R5: Emit each hj according to row i
        self.tmp_M5 = tmp_M5.map(lambda x: (x[1][1][0], x[1][1][1] * x[1][0]))

        
        
        #M6: Finish the computation of matrix S -- identity mapper so no code needed
        #R6: Emit each row in matrix S
        self.S = self.tmp_M5.reduceByKey(lambda x, y : x +y)

        
        #M7: Calculate matrix C (same as calculating B in compute_H)
        tmp_M7 = self.H.map(lambda x: (0, np.outer(x[1], x[1])))
        #R7: Emit matrix C from the sum of matrices
        C = tmp_M7.reduceByKey(lambda x, y : x +y).map(lambda x: x[1])
        
        #M8: Compute matrix T
        C_b = self.sc.broadcast(C.collect())
        self.T = self.W.map(lambda x: (x[0], x[1]@C_b.value))
        

        #M9: update W with broadcast join of S and T. W = W * S/T
        T_b = self.sc.broadcast(self.T.collectAsMap())
        S_b = self.sc.broadcast(self.S.collectAsMap())
        self.W = self.W.map(lambda x: (x[0], x[1]*np.divide(S_b.value[x[0]], T_b.value[x[0]])))
        
    def iterate_HW(self, max_iter=1): # , iteration_time=25, epsilon
        i = 0
        while i < max_iter: # or np.norm(previous_H, H) < epsilon
            print("Computing H and W for iteration " + str(i))
            self.compute_H()
            self.compute_W()
            i += 1
            
            
    def assign_clusters(self):
        print("Assign clusters")
        #M10: Map user to cluster with highest probability
        #self.clusters = self.H.map(lambda x: (x[0], x[1].index(max(x[1]))))
        self.clusters = self.H.map(lambda x: (x[0], np.where(x[1] == max(x[1]))[0][0]))
        
        #M11: Emit a 1 for each user that is in a cluster
        self.clustersizes = self.clusters.map(lambda x: (x[1], 1))
        
        #R11: Count the number of users per cluster
        self.clustersizes = self.clustersizes.reduceByKey(lambda x,y: x+y)
        
    def RM2_distribution(self):
        print("Compute RM2_distribution")
        #M12: Emit each rating that a user gave
        self.ratings = self.input.map(lambda x: (x[1], x[2]))
        
        #R12: Sum ratings given by user
        self.ratings = self.ratings.reduceByKey(lambda x, y: x+y)
        
        #Accumulator containing total count of ratings
        globalcount = self.sc.accumulator(0)
        self.ratings.foreach(lambda x: globalcount.add(x[1]))
        
        #M13: Emit each rating that an item received
        items = self.input.map(lambda x: (x[0], x[2]))
        
        #Broadcast the total count of ratings, to be used in next reduce
        globalcount = self.sc.broadcast(globalcount.value)
        
        #R13: Compute the probability of item in the collection
        items = items.reduceByKey(lambda x,y: x+y)
        
        #Divide previous values by the total count of ratings, to normalize probabilities
        items = items.map(lambda x: (x[0], x[1]/globalcount.value))
        
        #Broadcast all users and which cluster they are in
        clusters = self.sc.broadcast(self.clusters.collectAsMap())
        
        '''
        #M14-a: Map each rating from input to the cluster assigned to user
        self.input2cluster = self.input.map(lambda x: (clusters.value[x[1]], (x[1], x[0], x[2])))
        
        #M14-b: Map sum of ratings of a user to their assigned cluster
        self.ratings2cluster = self.ratings.map(lambda x: (clusters.value[x[0]], (x[0], x[1])))
        
        #Repartition join these two maps by cluster
        print('Repartition join these two maps by cluster')
        self.fullClusters = self.input2cluster.cogroup(self.ratings2cluster)
        '''
        #M14-a: Map each rating from input to the cluster assigned to user
        self.input2cluster = self.input.map(lambda x: (clusters.value[x[1]], (x[1], x[0], x[2]))).groupByKey()
        
        #M14-b: Map sum of ratings of a user to their assigned cluster
        self.ratings2cluster = self.ratings.map(lambda x: (clusters.value[x[0]], (x[0], x[1]))).groupByKey()
        
        #Repartition join these two maps by cluster
        self.fullClusters = self.input2cluster.join(self.ratings2cluster)
        
        #Broadcast sizes of the clusters
        #clustersizes = self.sc.broadcast(self.clustersizes.collectAsMap())
        
        #Broadcast probabilities of each item
        items_bc = self.sc.broadcast(items.collectAsMap())
        
        ld_bc = self.sc.broadcast(self.ld)
        
        #Final reduce combines all information from above, needs W and H
        self.out = self.fullClusters.flatMap(lambda x: (x[0], compute_p_i_R(x[1][0], x[1][1], items_bc.value, ld_bc.value)))
    
    def save_output(self):
        print("Saving the output to " + self.output_path)
        self.out.saveAsTextFile(self.output_path)
        print("Saved to " + self.output_path)
        self.sc.stop()
            
        
        
        

def main():
    data_files = '../data/input_data/train_data_*.txt'
    Algorithm = MRAlgorithm(data_files, '../data/output_data-k5', k=5, ld=0.5)
    #Algorithm.iterate_HW()
    #Algorithm.H.saveAsTextFile('../data/output_data-H')
    Algorithm.assign_clusters()
    Algorithm.RM2_distribution()
    #print('generating full clusters')
    #Algorithm.fullClusters.collect()
    #print('generating out')
    #Algorithm.out.collect()
    #Algorithm.out.saveAsTextFile('../data/output_data-k5')
    Algorithm.save_output()


if __name__ == '__main__':
    main()


