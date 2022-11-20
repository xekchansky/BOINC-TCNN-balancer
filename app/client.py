import argparse
import json
import os
import pathlib
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import NodeAPI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default=None)
    parser.add_argument('--port', type=int, required=False, default=12345)
    return parser.parse_args()


def main(ip, port):
    if ip is None:
        output_path = os.path.join(str(pathlib.Path(__file__).parent.parent.resolve()), 'terraform_output.json')
        with open(output_path) as f:
            ip = json.load(f)['external_ip_address_load_balancer']['value']

    NodeAPI(ip, port)


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port)
