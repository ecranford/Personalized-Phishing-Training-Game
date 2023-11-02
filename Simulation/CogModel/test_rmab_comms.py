####CODE FOR PYTHON TCP COMMUNICATION###
import json
import socketserver
##import random just for selecting random users, can remove if no longer using
import random

HOST = "127.0.0.1"
PORT = 9143

# DANGER, DANGER: This is pretty simple minded. It has no error handling or checking of
#                 data for the correct format. If anything isn't just as it's expected to
#                 be Bad Things will happen and there will be tears.

def get_users(user_data):
    ##currently returns two random users
    result = random.sample(sorted(user_data), k=2)
    return result   ##"("+result[0]+" "+result[1]+")"


class ExampleHandler (socketserver.StreamRequestHandler):

    def handle(self):
        user_data = self.rfile.readline().decode("utf-8").strip()
        print(f"# received message '{user_data}'")
        user_data = json.loads(user_data)
        print(f"# decoded JSON {user_data}")
        selected_users = get_users(user_data)
        print(f"# result is {selected_users}")
        selected_users = json.dumps(selected_users) #+ "\n"
        print(f"# sending '{selected_users}'")
        self.wfile.write(bytes(selected_users, "utf-8"))
        self.wfile.flush()


def open_port():
    with socketserver.TCPServer((HOST, PORT), ExampleHandler) as server:
        print(f"# opening port '{PORT}'")
        server.serve_forever()
        #server.handle_request()


if __name__== "__main__":
    open_port()
####CODE FOR PYTHON TCP COMMUNICATION###

###Parameters for running simulator.py
###python3 simulator.py -pc 44 -d simple_spam_ham -l 100 -s 0 -ws 0 -ls 0 -g 0.99 -S 3 -A 2 -n 1000 -lr 1 -N 1 -sts -1 -b 0.2