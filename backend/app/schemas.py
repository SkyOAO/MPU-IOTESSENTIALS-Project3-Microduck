from datetime import datetime
from pydantic import BaseModel,Field,model_validator

class CommandRequest(BaseModel):
    op: str | None = None
    action: str | None = Field(None, deprecated=True, description="已弃用，请改用 op")
    params: dict = Field(default_factory=dict)
    robot_id: str = "robot01"
    msg_id: str | None = None
    timestamp: int | None = None

    @model_validator(mode="after")
    def check_op(self):
        if not self.op:
            self.op = self.action
        if not self.op:
            raise ValueError("必须提供 op 或 action 字段")
        return self

class CommandResponse(BaseModel):
    command_id: str
    op: str
    action: str | None = None
    status: str

    @model_validator(mode="after")
    def fill_action(self):
        if self.action is None:
            self.action = self.op
        return self

class CommandStatus(BaseModel):
    command_id: str
    op: str
    action: str | None = None
    status: str
    created_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def fill_action(self):
        if self.action is None:
            self.action = self.op
        return self


class TelemetryLatest(BaseModel):
    robot_id: str | None = None
    ts: int | None = None
    received_at: datetime | None = None
    policy: str | None = None

class ConsoleStatus(BaseModel):
    status: str = "offline"
    current_behavior: str | None = None
    current_mode: str = "WALK"
    last_command: str | None = None
    timestamp: datetime | None = None
    joints: dict = Field(default_factory=dict)