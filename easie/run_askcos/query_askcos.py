from run_tree_builder_api2 import *

#csv file with targets in a column named 'smiles'
TARGETS_FILE = 'valsartan.csv'
#list of GCP VM instances to submit jobs to
HOSTS=['HOST URL']
#json files with parameters for tree builder
PARAMS_FILE = 'params.json'
PRIORITIZERS=[[{'template_set':'reaxys'}]]
SAVE_PREFIX='valsartan'

#run jobs in parallel split among the GCP VM instances
params = json.load(open(PARAMS_FILE,'r'))
targets = pd.read_csv(TARGETS_FILE).loc[:, 'smiles'].values
submit_parallel_job(targets, PRIORITIZERS, HOSTS, params, SAVE_PREFIX, return_results=False)
