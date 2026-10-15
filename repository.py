from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import Task, User


class TaskRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    @staticmethod
    def _tasks(telegram_user_id: int) -> Select[tuple[Task]]:
        return select(Task).join(User).where(User.telegram_id == telegram_user_id)

    async def get_or_create_user(self, telegram_user_id: int, username: str | None = None) -> User:
        async with self.sessions.begin() as session:
            statement = insert(User).values(telegram_id=telegram_user_id, username=username)
            if username is None:
                statement = statement.on_conflict_do_nothing(index_elements=[User.telegram_id])
            else:
                statement = statement.on_conflict_do_update(
                    index_elements=[User.telegram_id], set_={"username": username}
                )
            await session.execute(statement)
            return (
                await session.scalars(select(User).where(User.telegram_id == telegram_user_id))
            ).one()

    async def create_task(self, telegram_user_id: int, **values: Any) -> Task:
        user = await self.get_or_create_user(telegram_user_id)
        async with self.sessions.begin() as session:
            task = Task(user_id=user.id, **values)
            session.add(task)
            await session.flush()
            await session.refresh(task)
            return task

    async def get_task(self, telegram_user_id: int, task_id: int) -> Task | None:
        async with self.sessions() as session:
            return await session.scalar(self._tasks(telegram_user_id).where(Task.id == task_id))

    async def get_tasks(
        self,
        telegram_user_id: int,
        *,
        status: str | None = "pending",
        priority: str | None = None,
        deadline_from: datetime | None = None,
        deadline_to: datetime | None = None,
    ) -> list[Task]:
        statement = self._tasks(telegram_user_id)
        if status is not None:
            statement = statement.where(Task.status == status)
        if priority is not None:
            statement = statement.where(Task.priority == priority)
        if deadline_from is not None:
            statement = statement.where(Task.deadline >= deadline_from)
        if deadline_to is not None:
            statement = statement.where(Task.deadline < deadline_to)
        async with self.sessions() as session:
            return list(
                await session.scalars(statement.order_by(Task.deadline.asc().nulls_last(), Task.id))
            )

    async def find_tasks_by_title(self, telegram_user_id: int, query: str) -> list[Task]:
        async with self.sessions() as session:
            statement = self._tasks(telegram_user_id)
            exact = list(
                await session.scalars(statement.where(func.lower(Task.title) == query.lower()))
            )
            if exact:
                return exact
            statement = statement.where(Task.title.icontains(query, autoescape=True)).order_by(
                Task.id
            )
            return list(await session.scalars(statement))

    async def update_task(self, telegram_user_id: int, task_id: int, **changes: Any) -> Task | None:
        async with self.sessions.begin() as session:
            task = await session.scalar(
                self._tasks(telegram_user_id).where(Task.id == task_id).with_for_update()
            )
            if task is None:
                return None
            for field, value in changes.items():
                setattr(task, field, value)
            await session.flush()
            await session.refresh(task)
            return task

    async def complete_task(self, telegram_user_id: int, task_id: int) -> Task | None:
        return await self.update_task(telegram_user_id, task_id, status="completed")

    async def delete_task(self, telegram_user_id: int, task_id: int) -> bool:
        async with self.sessions.begin() as session:
            task = await session.scalar(
                self._tasks(telegram_user_id).where(Task.id == task_id).with_for_update()
            )
            if task is None:
                return False
            await session.delete(task)
            return True

    async def get_pending_reminders(self, now: datetime) -> list[tuple[int, Task]]:
        async with self.sessions() as session:
            rows = await session.execute(
                select(User.telegram_id, Task)
                .join(Task)
                .where(
                    Task.status == "pending",
                    Task.reminder_sent.is_(False),
                    Task.next_reminder_at <= now,
                )
                .order_by(Task.next_reminder_at, Task.id)
                .limit(100)
            )
            return [(telegram_id, task) for telegram_id, task in rows]

    async def get_unscheduled_reminders(self) -> list[tuple[int, Task]]:
        async with self.sessions() as session:
            rows = await session.execute(
                select(User.telegram_id, Task)
                .join(Task)
                .where(
                    Task.status == "pending",
                    Task.reminder_sent.is_(False),
                    Task.deadline.is_not(None),
                    Task.next_reminder_at.is_(None),
                )
                .order_by(Task.id)
                .limit(100)
            )
            return [(telegram_id, task) for telegram_id, task in rows]

    async def advance_reminder(
        self,
        telegram_user_id: int,
        task_id: int,
        deadline: datetime,
        expected_reminder_at: datetime | None,
        next_reminder_at: datetime | None,
    ) -> None:
        async with self.sessions.begin() as session:
            task = await session.scalar(
                self._tasks(telegram_user_id).where(Task.id == task_id).with_for_update()
            )
            if (
                task is not None
                and task.status == "pending"
                and task.deadline == deadline
                and task.next_reminder_at == expected_reminder_at
            ):
                task.next_reminder_at = next_reminder_at
                task.reminder_sent = next_reminder_at is None
