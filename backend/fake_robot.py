import json
import time
import threading
import paho.mqtt.client as mqtt

BROKER = "8.138.114.112"
PORT = 1883
TOPIC = "microduck/robot01/cmd"
ACK_TOPIC = "microduck/robot01/ack"
TLM_TOPIC = "microduck/robot01/telemetry"
HZ = 50

_current_op = "idle"
_tlm_started = threading.Event()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("连上了，reason_code =", reason_code, flush=True)
    result, mid = client.subscribe(TOPIC, qos=1)
    print("订阅", TOPIC, "-> result =", result, flush=True)

    if not _tlm_started.is_set():
        _tlm_started.set()
        threading.Thread(target=telemetry_loop, args=(client,), daemon=True).start()


def send_ack(client, msg_id, phase, result=None):
    ack = {
        "msg_id": msg_id,
        "robot_id": "robot01",
        "phase": phase,
        "result": result,
        "ts": int(time.time() * 1000),
    }
    client.publish(ACK_TOPIC, json.dumps(ack), qos=1)
    print("已回 ack ->", phase, flush=True)


def telemetry_loop(client):
    period = 1.0 / HZ
    next_t = time.perf_counter()
    n = 0
    while True:
        next_t += period
        payload = {
            "robot_id": "robot01",
            "ts": int(time.time() * 1000),
            "policy": _current_op,
            "trunk_z": 0.212,
        }
        client.publish(TLM_TOPIC, json.dumps(payload), qos=0)
        n += 1
        if n % 50 == 0:
            print("已发遥测", n, "条", flush=True)

        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()


def do_action(client, msg_id, op):
    global _current_op
    _current_op = op
    send_ack(client, msg_id, "received")
    time.sleep(0.5)
    send_ack(client, msg_id, "started")
    time.sleep(2)
    send_ack(client, msg_id, "finished", result="ok")
    _current_op = "idle"
    print("动作完成:", op, flush=True)
    print("-" * 50, flush=True)


def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    op = data.get("op") or data.get("action")
    msg_id = data.get("msg_id")
    print("收到命令 op =", op, "msg_id =", msg_id, flush=True)

    t = threading.Thread(target=do_action, args=(client, msg_id, op), daemon=True)
    t.start()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()