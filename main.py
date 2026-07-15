# main.py
import threading
import time
import uuid

import listener
import uvicorn


def run_api():
    uvicorn.run("api:app", host="0.0.0.0", port=8000)


def run_listener():
    time.sleep(2)  # attend que l'API soit prete
    listener.main(run_id=uuid.uuid4().hex)


if __name__ == "__main__":
    t1 = threading.Thread(target=run_api)
    t2 = threading.Thread(target=run_listener)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
