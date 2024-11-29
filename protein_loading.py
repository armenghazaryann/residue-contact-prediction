from URLs import *

import os

import requests

def query_payload(num: int, start=1) -> dict[str, str | dict[str, str | dict[str, str]] | dict[str, dict[str, int]]]:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match",
                "value": "Protein (only)"
            }
        },
        "request_options": {
            "paginate": {
                "start": start,
                "rows": num
            }
        },
        "return_type": "entry"
    }


def get_protein_ids(url: str, number_of_proteins: int) -> list:
    _response = requests.post(url, json=query_payload(number_of_proteins, start=1))
    if _response.status_code == 200:
        _result = _response.json()
        _protein_ids = [item["identifier"] for item in _result.get("result_set", [])]
    else:
        raise Exception(f"Failed to fetch data: {_response.status_code}, {_response.text}")
    return _protein_ids

def download_pdb_files(url: str, protein_ids: list, output_folder: str="pdb_files") -> None:
    os.makedirs(output_folder, exist_ok=True)
    for protein_id in protein_ids:
        if not os.path.exists(f"{output_folder}/{protein_id}.pdb"):
            _response = requests.get(f"{url}/{protein_id}.pdb")
            if _response.status_code == 200:
                with open(os.path.join(output_folder, f"{protein_id}.pdb"), "wb") as file:
                    file.write(_response.content)
            else:
                print(f"Failed to download: {protein_id}.pdb")
        else:
            print(f"{protein_id} is already downloaded")

protein_ids=get_protein_ids(SearchURL, 100)
download_pdb_files(DownloadURL, protein_ids)
