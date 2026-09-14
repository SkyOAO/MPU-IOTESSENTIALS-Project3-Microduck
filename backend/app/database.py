from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,declarative_base
from backend.app.config import DATABASE_URL

# 建立物理连接
engine = create_engine(DATABASE_URL)

# 创建会话工厂
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# 创建模型基类
Base = declarative_base()

# 分配窗口给fastapi
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 方法:
# 增：db.add(对象) + db.commit()
#
# 查：db.query(模型).filter(条件).first()
#
# 改：查到对象 → 改属性 → db.commit()
#
# 删：查到对象 → db.delete(对象) + db.commit()