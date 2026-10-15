from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    inspect,
    text,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    tasks: Mapped[list["Task"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint("priority IN ('low', 'medium', 'high')"),
        CheckConstraint("status IN ('pending', 'completed')"),
        Index("ix_tasks_user_status", "user_id", "status"),
        Index("ix_tasks_reminders", "status", "reminder_sent", "deadline"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    priority: Mapped[str] = mapped_column(String(10), default="medium")
    status: Mapped[str] = mapped_column(String(10), default="pending")
    reminder_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    next_reminder_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    user: Mapped[User] = relationship(back_populates="tasks")


class Database:
    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url, pool_pre_ping=True, hide_parameters=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            columns = await connection.run_sync(
                lambda sync: {column["name"] for column in inspect(sync).get_columns("tasks")}
            )
            if "next_reminder_at" not in columns:
                await connection.execute(
                    text("ALTER TABLE tasks ADD COLUMN next_reminder_at TIMESTAMP WITH TIME ZONE")
                )
                await connection.execute(
                    text(
                        "UPDATE tasks SET reminder_sent = false "
                        "WHERE status = 'pending' AND deadline > CURRENT_TIMESTAMP"
                    )
                )
            await connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_tasks_next_reminder "
                    "ON tasks (next_reminder_at) WHERE status = 'pending' AND reminder_sent = false"
                )
            )

    async def close(self) -> None:
        await self.engine.dispose()
