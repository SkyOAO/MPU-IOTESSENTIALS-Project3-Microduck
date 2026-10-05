import json
import logging
from datetime import datetime,timezone

from app import models
from app.database import SessionLocal

logger = logging.getLogger(__name__)

ACK_TOPIC = "microduck/robot01/ack"
TELEMETRY_TOPIC = "microduck/robot01/telemetry"

async def consume(client):
    await client.subscribe(ACK_TOPIC, qos=1)
    await client.subscribe(TELEMETRY_TOPIC, qos=0)
    print(f"Connected to {ACK_TOPIC}", flush=True)
    print(f"Connected to {TELEMETRY_TOPIC}", flush=True)

    async for message in client.messages:
        topic = str(message.topic)
        try:
            data = json.loads(message.payload)
        except Exception as e:
            logger.warning("[UPLINK] bad json on %s: %s", topic, e)
            continue

        if topic.endswith("/ack"):
            handle_ack(data)
        elif topic.endswith("/telemetry"):
            handle_telemetry(data)




def handle_ack(data:dict) -> None:
    msg_id = data.get("msg_id") or data.get("command_id")
    phase = data.get("phase")
    result = data.get("result")

    try:
        with SessionLocal() as db:
            db.add(models.CommandAck(
                command_id=msg_id,
                robot_id=data.get("robot_id"),
                phase=phase,
                result=result,
                detail=data.get("detail"),
                robot_ts=data.get("ts"),
            ))

            cmd = db.query(models.Command).filter_by(command_id=msg_id).first()
            if cmd is not None:
                if phase == "received":
                    cmd.status = "received"
                elif phase == "started":
                    cmd.status = "started"
                elif phase == "finished":
                    cmd.status = "finished" if result == "ok" else "failed"
                    cmd.completed_at = datetime.now(timezone.utc)

            db.commit()
            print(f"ACK {msg_id} phase= {phase} result={result}"
                  f" status={cmd.status if cmd else 'NOT FOUND'}",flush=True)

    except Exception as e:
        logger.exception("[UPLINK] handle_ack failed: %s", e)

def handle_telemetry(data: dict) -> None:
    try:
        with SessionLocal() as db:
            db.add(models.Telemetry(
                robot_id=data.get("robot_id"),
                ts=data.get("ts"),
                policy=data.get("policy"),
                payload=data,
            ))
            db.commit()
    except Exception as e:
        logger.exception("[UPLINK] handle_telemetry failed: %s", e)