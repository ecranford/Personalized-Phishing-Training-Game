# Copyright (c) 2022 Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import math
import socketserver

HOST = "127.0.0.1"
PORT = 9893

# DANGER, DANGER: This is pretty simple minded. It has no error handling or checking of
#                 data for the correct format. If anything isn't just as it's expected to
#                 be Bad Things will happen and there will be tears.

def run(data):
    # result = "-".join([data["word"]] * int(data["repeat"]))
    round = [data["round"]] + 1
    # return {"string": result, "x": math.log(len(result))}
    return {"round": round, "action": {"user1": random.randint(0, 1), "user2": random.randint(0, 1)}}


class ExampleHandler (socketserver.StreamRequestHandler):

    def handle(self):
        data = self.rfile.readline().decode("utf-8").strip()
        print(f"# received message '{data}'")
        data = json.loads(data)
        print(f"# decoded JSON {data}")
        data = run(data)
        print(f"# result is {data}")
        data = json.dumps(data) + "\n"
        print(f"# sending '{data}'")
        self.wfile.write(bytes(data, "utf-8"))
        self.wfile.flush()


def main():
    with socketserver.TCPServer((HOST, PORT), ExampleHandler) as server:
        server.handle_request()


if __name__== "__main__":
    main()
