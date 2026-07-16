from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings

engine = create_async_engine(f"sqlite+aiosqlite:///{settings.storage_dir}/chat.db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(default="New chat")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    role: Mapped[str]  # 'user' | 'assistant'
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    blocks: Mapped[list["Block"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="Block.position",
    )


class Block(Base):
    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    type: Mapped[str]  # 'thought' | 'tool' | 'answer' | 'user-query'
    content: Mapped[str] = mapped_column(Text)  # tool -> json.dumps(hits)
    position: Mapped[int]

    message: Mapped["Message"] = relationship(back_populates="blocks")


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
