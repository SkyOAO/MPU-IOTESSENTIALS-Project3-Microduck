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
    status: str