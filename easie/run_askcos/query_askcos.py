from run_tree_builder_api2 import *

#csv file with targets in a column named 'smiles'
TARGETS_FILE = 'jnk3_hits_example.csv'
#list of ASKCOS instance urls to submit jobs to
#requires spinning up ASKCOS instances. See: https://github.com/ASKCOS
HOSTS=['']
#json files with parameters for tree builder
PARAMS_FILE = 'params.json'
PRIORITIZERS=[[{'template_set':'reaxys'}]]
SAVE_PREFIX='jnk-3_hits'

#run jobs in parallel split among the GCP VM instances
params = json.load(open(PARAMS_FILE,'r'))
targets = pd.read_csv(TARGETS_FILE).loc[:, 'smiles'].values
submit_parallel_job(targets, PRIORITIZERS, HOSTS, params, SAVE_PREFIX, return_results=False)
