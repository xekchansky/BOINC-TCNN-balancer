import argparse
import pathlib
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import LoadBalancerAPI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default='')
    parser.add_argument('--port', type=int, required=False, default=12345)
    return parser.parse_args()


def main(ip, port):
    ip = ip
    port = port

    LoadBalancerAPI(ip, port)


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port)
