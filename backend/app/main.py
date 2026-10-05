import json
import os
from datetime import datetime, timezone
import time
import uuid
import asyncio
from contextlib import asynccontextmanager

import aiomqtt
from fastapi import FastAPI,Depends,Request,HTTPException
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.database import engine,Base,get_db
from app import models
from app import uplink
from app.schemas import CommandRequest,CommandResponse,CommandStatus,TelemetryLatest,ConsoleStatus




from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):

    async with aiomqtt.Client(
            hostname="localhost",
            port=1883,
            username=os.getenv("MQTT_USER"),
            password=os.getenv("MQTT_PASS"),
    ) as client:

        app.state.mqtt = client
        uplink_task = asyncio.create_task(uplink.consume(client))
        # noinspection PyUnresolvedReferences
        app.state.mqtt = client
        print("MQTT connected",flush=True)
        yield
        uplink_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- API Key 鉴权 ----------
API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Depends(_api_key_header)):
    """校验请求头 X-API-Key。未配置 API_KEY 时（本地开发）不校验。"""
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"message": "Cloud backend is running"}

@app.post("/api/v1/commands", dependencies=[Depends(verify_api_key)])
async def create_command(request: Request, command: CommandRequest,db: Session = Depends(get_db)):

    command_id = str(uuid.uuid4())

    new_item = models.Command(
        command_id=command_id,
        action=command.op,
        status="received",
    )
    db.add(new_item)
    db.commit()

    client = request.app.state.mqtt
    await client.publish(
        "microduck/robot01/cmd",
        json.dumps({
            "msg_id": command_id,
            "robot_id": command.robot_id,
            "op": command.op,
            "params": command.params,
            "timestamp": int(time.time() * 1000),        # 毫秒，后端生成
        }),
        qos=1,                                        # 至少送达一次
    )

    return CommandResponse(
        command_id=command_id,
        op=command.op,
        status="received",
    )

@app.get("/api/v1/commands", response_model=list[CommandStatus], dependencies=[Depends(verify_api_key)])
def list_commands(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(models.Command)
        .order_by(models.Command.id.desc())
        .limit(limit)
        .all()
    )
    return [
        CommandStatus(
            command_id=r.command_id,
            op=r.action,
            status=r.status,
            created_at=r.created_at,
            completed_at=r.completed_at,
        )
        for r in rows
    ]


@app.get("/api/v1/commands/{command_id}", response_model=CommandStatus, dependencies=[Depends(verify_api_key)])
def get_command(command_id: str, db: Session = Depends(get_db)):
    r = db.query(models.Command).filter_by(command_id=command_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail="command not found")
    return CommandStatus(
        command_id=r.command_id,
        op=r.action,
        status=r.status,
        created_at=r.created_at,
        completed_at=r.completed_at,
    )


@app.get("/api/v1/telemetry/latest", response_model=TelemetryLatest, dependencies=[Depends(verify_api_key)])
def latest_telemetry(db: Session = Depends(get_db)):
    r = (
        db.query(models.Telemetry)
        .order_by(models.Telemetry.id.desc())
        .first()
    )
    if r is None:
        return TelemetryLatest()
    return TelemetryLatest(
        robot_id=r.robot_id,
        ts=r.ts,
        received_at=r.received_at,
        policy=r.policy,
    )

def _console_status(db: Session) -> ConsoleStatus:
    last_cmd = (
        db.query(models.Command)
        .order_by(models.Command.id.desc())
        .first()
    )
    last_tlm = (
        db.query(models.Telemetry)
        .order_by(models.Telemetry.id.desc())
        .first()
    )

    status = "offline"
    behavior = None
    ts = None
    joints = {}

    if last_tlm is not None:
        ts = last_tlm.received_at
        behavior = last_tlm.policy
        payload = last_tlm.payload if isinstance(last_tlm.payload, dict) else {}
        if isinstance(payload.get("joints"), dict):
            joints = payload["joints"]

        if last_tlm.received_at is not None:
            got = last_tlm.received_at
            if got.tzinfo is None:
                got = got.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - got
            if age.total_seconds() <= 10:
                status = "online"

    return ConsoleStatus(
        status=status,
        current_behavior=behavior,
        last_command=last_cmd.action if last_cmd is not None else None,
        timestamp=ts,
        joints=joints,
    )


@app.get("/api/v1/status", response_model=ConsoleStatus, dependencies=[Depends(verify_api_key)])
def get_status(db: Session = Depends(get_db)):
    return _console_status(db)


@app.get("/api/v1/telemetry", response_model=ConsoleStatus, dependencies=[Depends(verify_api_key)])
def get_telemetry_console(db: Session = Depends(get_db)):
    return _console_status(db)
# @app.get("/test-db")
# def test_db(db: Session = Depends(get_db)):
#     item = models.TempTest(note="管道测试")
#     db.add(item)
#     db.commit()
#     db.refresh(item)
#     return {"id": item.id, "note": item.note}

# document.getElementById('danceButton').addEventListener('click', () => {
#     // 当按下按钮，网页执行 fetch，发请求给后端
#     fetch('http://你的电脑IP:8000/api/v1/commands', {
#         method: 'POST',
#         body: JSON.stringify({"action": "dance"})
#     })
# })
