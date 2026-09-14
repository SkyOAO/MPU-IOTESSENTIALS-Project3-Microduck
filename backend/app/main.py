import uuid
import aiomqtt
import json

from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from backend.app.database import engine,Base,get_db
from backend.app import models
from backend.app.schemas import CommandRequest,CommandResponse




from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Cloud backend is running"}

@app.post("/api/v1/commands")
async def create_command(command: CommandRequest,db: Session = Depends(get_db)):

    command_id = str(uuid.uuid4())

    new_item = models.Command(
        command_id=command_id,
        action=command.action,
        status="received",
    )
    db.add(new_item)
    db.commit()

    async with aiomqtt.Client(hostname="localhost", port=1883) as client:
        await client.publish(
            "microduck/robot01/cmd",
            json.dumps({
                "command_id": command_id,
                "action": command.action
            })
        )

    return CommandResponse(
        command_id=command_id,
        action=command.action,
        status="received",
    )


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
