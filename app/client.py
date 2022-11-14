from utils.api import NodeAPI


def main():
    # config = configparser.ConfigParser()

    ip = 'localhost'
    port = 12345

    client = NodeAPI(ip, port)


if __name__ == "__main__":
    main()
