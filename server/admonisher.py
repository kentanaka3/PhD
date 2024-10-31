import dask.distributed as dd

cluster = dd.LocalCluster()

client = dd.Client(cluster)