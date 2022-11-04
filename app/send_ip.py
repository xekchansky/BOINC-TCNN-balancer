import smtplib, ssl
import socket
import configparser

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

port = 465  # For SSL

# Create a secure SSL context
context = ssl.create_default_context()

config = configparser.ConfigParser()
config.read('config.ini')
acc = config['gmail_acc']

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(acc['email'], acc['password'])
    message = """\
Subject: Worker is Ready
""" + str(extract_ip())
    server.sendmail(acc['email'], "xekchansky@gmail.com", message)
