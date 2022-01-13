# This script reads results from number_of_deltas-mse experiment, just run it

import neptune.new as neptune
# pick which experiments you want to see (by tag)
TAG = 'deltas-mse'

################################
project = neptune.get_project(
    name="rm360179/lidl-loss-mse",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==")
run_table = project.fetch_runs_table(
        tag=TAG).to_pandas()

ids_with_tag = [str(x) for x in run_table['sys/id']]

deltas = list()
mse = list()
for run_id in ids_with_tag:
    run = neptune.init(
        mode='read-only',
        project="rm360179/lidl-loss-mse",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==",
        run=run_id)
    deltas.append(run['num_deltas'].fetch())
    mse.append(run['mse'].fetch())
    run.stop()

print(deltas)
print(mse)
