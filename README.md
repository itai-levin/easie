## EASIE - Exploration of chemical Analog Space, Implicitly and Explicitly

### Usage:
To download the ASKCOS building block database:

1. `cd easie/building_blocks`
2. `wget https://github.com/ASKCOS/askcos-data/blob/main/buyables/buyables.json.gz`

To count the number of analogs or perform an explicit enumeration for a manually defined route, use `count_analogs.py` or `enumerate_analogs.py`, respectively. 
Manually defined routes should be written into a file as a list of reaction SMILES with double quotation marks, separated by commas. The order of the reactions does not matter.

To count the number of analogs or perform an explicit enumeration for routes generated by ASKCOS use `count_analogs_askcos_route.py` or `enumerate_analogs_askcos_route.py`, respectively. 
To generate routes from ASKCOS in the appropariate format, use the script `easie/run_askcos/query_askcos.py`. Example input files are provided in the `easie/run_askcos` directory. This will require a separate deployment of [ASKCOS](https://github.com/ASKCOS).


Authors: Itai Levin and Michael Fortunato
