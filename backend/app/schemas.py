from pydantic import BaseModel

class CommandRequest(BaseModel):
    action: str

class CommandResponse(BaseModel):
    command_id: str
    action: str
    status: str