

These tests are created to ensure correctness as I progress towards my implementation v smoothly IA. 


Test # 1 (Verifying the Correctness of the Snapshotting and Restore):
--> Over here we run sample test on simple configuration for few number of requests and then compare the results in the batch metrics csv and the request metrics csv generated in the vidur/simulator output. 
--> Note for batch metrics we compare the results for the batches that occured after the snapshots.
--> For the Request metrics we compare the results : We first combine the requests that were present in the post snapshot batch metrics.csv and then compare their request metrics. Note that since we are not snapshotting metrics independently, we have to drop some metrics. Nevertheless, the resultant comparison is enough to ensure that state of simulator is preserved throughout







