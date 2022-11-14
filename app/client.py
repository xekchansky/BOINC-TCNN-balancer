from utils.api import NodeAPI


def main():
    # config = configparser.ConfigParser()

    ip = '158.160.39.178'
    port = 12345

    NodeAPI(ip, port)


if __name__ == "__main__":
    main()
