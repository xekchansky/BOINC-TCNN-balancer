import sys
import pathlib
sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import LoadBalancerAPI


def main():
    ip = ''#'0.0.0.0'
    port = 12345

    LoadBalancerAPI(ip, port)


if __name__ == "__main__":
    main()
