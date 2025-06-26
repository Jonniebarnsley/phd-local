import os
import ssl
import requests
from tqdm import tqdm
from pathlib import Path
from ssl import SSLContext
from itertools import product
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection
from pyesgf.search.results import DatasetResult, FileResult
from requests.exceptions import RequestException

# --- Configuration ---

# Ensure that these environment variables are set, e.g. in your .bashrc
USERNAME = os.environ.get('ESGF_USERNAME')
PASSWORD = os.environ.get('ESGF_PASSWORD')
DATA_PATH = Path(os.environ.get('DATA_HOME', './')) # Default to current directory if not set

MYPROXY_HOST = 'esgf-node.ipsl.upmc.fr'

'''
N.B. Login to ESGF can be very temperamental. If you're having issues, try creating an
account on one of the other ESGF nodes, such as:

    esgf.ceda.ac.uk
    esgf-data.dkrz.de
    esgf-node.ipsl.upmc.fr

And setting this as MYPROXY_HOST with the correct username and password (username and password
may differ between nodes). Logging in this way will still allow you to access data from all
ESGF nodes. You should still download from the node geographically closest to you for speed. 
'''

SEARCH_NODE = 'http://esgf-node.ipsl.upmc.fr/esg-search'

# You should ideally set this to the ESGF node geographically closest to you
DATA_NODE_PREFERENCE = 'esgf.ceda.ac.uk'

PROJECT = 'CMIP6'
FREQUENCY = 'mon'   # monthly 
GRID_LABEL = 'gn'   # gr - regridded to lat-lon, gn - native grid

SCENARIOS = ['historical', 'ssp126', 'ssp585']
VARIABLES = ['tas', 'pr', 'evspsbl', 'mrro', 'thetao', 'so']
MODELS = ['UKESM1-0-LL', 'CESM2-WACCM', 'IPSL-CM6A-LR', 'ACCESS-ESM1-5', 'MRI-ESM2-0', \
          'CanESM5']#, 'CNRM-ESM2-1', 'MIROC-ES2L', 'ACCESS-CM2']

TABLE_ID = {
    'tas': 'Amon',
    'pr': 'Amon',
    'evspsbl': 'Amon',
    'mrro': 'Lmon',
    'thetao': 'Omon',
    'so': 'Omon'
}


# UKESM1-0-LL has a different variant label to most models
VARIANT_LABEL = {model: 'r4i1p1f2' if model == 'UKESM1-0-LL' else 'r1i1p1f1' for model in MODELS}


# --- Main ---

def login_to_esgf(username, password, hostname) -> SSLContext:

    '''
    N.B. Login to ESGF can be very temperamental. If you're having issues, try creating an
    account on one of the other ESGF nodes, such as:

        esgf.ceda.ac.uk
        esgf-data.dkrz.de
        esgf-node.ipsl.upmc.fr

    And setting this as myproxy_host with the correct username and password (username and password
    may differ between nodes). Logging in this way will still allow you to access data from all
    ESGF nodes. You should still download from the node geographically closest to you for speed. 
    '''

    if not username or not password:
        raise ValueError("ESGF_USERNAME and ESGF_PASSWORD environment variables must be set.")
    
    lm = LogonManager()
    if not lm.is_logged_on():
        lm.logon(username=username, password=password, hostname=hostname)
        print(f"Successfully logged into ESGF node: {hostname}")

    sslcontext = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    sslcontext.load_verify_locations(capath=lm.esgf_certs_dir)
    sslcontext.load_cert_chain(lm.esgf_credentials)
    return sslcontext

def get_local_path(dataset: DatasetResult, data_home: Path) -> Path:

    '''
    Generates a local path for the dataset based on its ID.
    The path is structured as:
    <data_home>/<project>/<activity>/<institute>/<model>/<experiment>/<realisation>/
        <time_resolution>/<variable>/gr/<version>/
    where <data_home> is the path to your local data directory.
    '''

    id, node = dataset.dataset_id.split('|')
    identifiers = id.split('.')
    path = data_home / Path(*identifiers)

    return path

def download_file(file: FileResult, savedir: Path) -> None:

    '''
    Downloads a single file to a local directory.
    '''

    url = file.download_url
    filename = file.filename
    filepath = savedir / filename
    if os.path.isfile(filepath):
        print(f"File {filename} already exists, skipping download.")
        return
    
    try:
        response = requests.get(file.download_url, stream=True, timeout=30)
        response.raise_for_status() # Raise an error on bad status
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192

        with open(filepath, 'wb') as f, tqdm(
            desc=file.filename,
            total=total_size,
            unit='B',          # Unit is Bytes
            unit_scale=True,   # Automatically convert to KB, MB, etc.
            unit_divisor=1024, # Use 1024 for conversion
            leave=True         # Leave the completed bar in the console
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    # Update the progress bar by the size of the chunk written
                    pbar.update(len(chunk)) 

    except RequestException as e:
        print(f"Failed to download {file.filename}: {e}")
        if filepath.exists(): # Clean up partial downloads
             os.remove(filepath)
    
    except KeyboardInterrupt:
        print("Download interrupted by user.")
        if filepath.exists():
            os.remove(filepath)
        raise


def download_dataset(dataset: DatasetResult) -> None:

    '''
    Downloads all files in a dataset to a local directory.
    The local directory is created based on the dataset ID.
    '''
    
    savedir = get_local_path(dataset, DATA_PATH)
    savedir.mkdir(parents=True, exist_ok=True)

    files = dataset.file_context().search(ignore_facet_check=True)
    for file in files:
        download_file(file, savedir)

def main() -> None:

    try:
        ssl_context = login_to_esgf(USERNAME, PASSWORD, MYPROXY_HOST)
    except Exception as e:
        print(f"ESGF login failed: {e}")
        return

    conn = SearchConnection(SEARCH_NODE, distrib=True)
    for scenario, model, variable in product(SCENARIOS, MODELS, VARIABLES):

        query = {
            'project': PROJECT,
            'source_id': model,
            'variant_label': VARIANT_LABEL[model],
            'experiment_id': scenario,
            'variable': variable,
            'table_id': TABLE_ID[variable],
            'frequency': FREQUENCY,
            'variant_label': VARIANT_LABEL[model],
            #'version': VERSION[model][scenario],
            'data_node': DATA_NODE_PREFERENCE,
            'grid_label': GRID_LABEL,
        }

        context = conn.new_context(**query, facets=query.keys())

        if context.hit_count == 0:
            print(f"No datasets found for {scenario}, {model}, {variable}.")
            continue

        results = context.search()
        dataset = list(results)[-1] # most recent dataset version

        download_dataset(dataset)

if __name__ == "__main__":
    main()
    print("Download complete.")