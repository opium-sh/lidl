import neptune.new as neptune
import numpy as np
# pick which experiments you want to see (by tag)
TAG = 'chart'

################################
project = neptune.get_project(
    name="rm360179/lidl-loss-mse",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==")
run_table = project.fetch_runs_table(
        tag=TAG).to_pandas()

ids_with_tag = [str(x) for x in run_table['sys/id']]


## This loop goes through all experiments that have "chart" tag
for run_id in ids_with_tag:
    run = neptune.init(
        mode='read-only',
        project="rm360179/lidl-loss-mse",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==",
        run=run_id)
    ##### HERE U CAN READ WHAT YOU NEED USING run['fieldname'].fetch() for single values and
    ##### run['fieldname'].fetch_values() for series 
    algorithm = run['algorithm'].fetch()
    dataset = run['dataset'].fetch()
    seed = run['seed'].fetch()
    lids = run['lids'].fetch_values()


    filename = f'{dataset}_{algorithm}_{seed}.csv'
    nplids = lids.value.to_numpy()
    np.savetxt(filename, nplids, delimiter=",")
