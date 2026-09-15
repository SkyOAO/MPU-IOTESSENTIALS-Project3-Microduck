from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime,timezone
from app.database import Base
from datetime import datetime, timezone, timedelta

CN_TZ = timezone(timedelta(hours=8))

class Command(Base):
    __tablename__ = 'commands'

    #id 自增整数主键
    id = Column(Integer, primary_key=True, index=True)
    #命令id 字符串 不重复 索引
    command_id = Column(String, unique=True, index=True)
    #动作
    action = Column(String)
    #状态
    status = Column(String)
    #创建时间
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    #完成时间
    completed_at = Column(DateTime(timezone=True), nullable=True)



