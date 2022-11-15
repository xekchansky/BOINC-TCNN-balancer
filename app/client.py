import json
import pathlib
import os
import sys
sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import NodeAPI


def main():
    # config = configparser.ConfigParser()
    output_path = os.path.join(str(pathlib.Path(__file__).parent.parent.resolve()), 'terraform_output.json')
    with open(output_path) as f:
        data = json.load(f)

    ip = data['external_ip_address_load_balancer']['value']
    port = 12345

    NodeAPI(ip, port)


if __name__ == "__main__":
    main()
