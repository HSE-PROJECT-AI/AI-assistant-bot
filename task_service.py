from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from database import Task
from repository import TaskRepository

Priority = Literal["low", "medium", "high"]
Status = Literal["pending", "completed"]
Title = Annotated[str, Field(min_length=1, max_length=200)]
Description = Annotated[str, Field(max_length=2000)]


def next_reminder_at(deadline: datetime | None, now: datetime) -> datetime | None:
    if deadline is None:
        return None
    remaining = deadline - now
    days = remaining.days
    if days >= 1:
        candidate = deadline - timedelta(days=days)
        if candidate > now:
            return candidate
        if days > 1:
            return deadline - timedelta(days=days - 1)
    for minutes in (720, 180, 60, 15):
        candidate = deadline - timedelta(minutes=minutes)
        if candidate > now:
            return candidate
    return None


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Use an ISO 8601 datetime with timezone") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime must include a timezone offset")
    return value.astimezone(UTC)


Deadline = Annotated[datetime | None, BeforeValidator(parse_datetime)]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class CreateTask(ToolInput):
    title: Title
    description: Description | None = None
    deadline: Deadline = None
    priority: Priority = "medium"


class TaskFilter(ToolInput):
    status: Status | None = "pending"
    priority: Priority | None = None
    deadline_from: Deadline = Field(
        default=None, description="Inclusive start, ISO 8601 with offset"
    )
    deadline_to: Deadline = Field(default=None, description="Exclusive end, ISO 8601 with offset")

    @model_validator(mode="after")
    def validate_range(self) -> "TaskFilter":
        if self.deadline_from and self.deadline_to and self.deadline_from >= self.deadline_to:
            raise ValueError("deadline_from must be before deadline_to")
        return self


class TaskSelector(ToolInput):
    task_id: Annotated[int, Field(gt=0)] | None = None
    query: Title | None = Field(default=None, description="Task title or a part of its title")

    @model_validator(mode="after")
    def validate_selector(self) -> "TaskSelector":
        if (self.task_id is None) == (self.query is None):
            raise ValueError("Provide exactly one of task_id or query")
        return self


class UpdateTask(TaskSelector):
    title: Title | None = None
    description: Description | None = None
    deadline: Deadline = None
    priority: Priority | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> "UpdateTask":
        fields = self.model_fields_set - {"task_id", "query"}
        if not fields:
            raise ValueError("Provide at least one field to update")
        if ("title" in fields and self.title is None) or (
            "priority" in fields and self.priority is None
        ):
            raise ValueError("title and priority cannot be null")
        return self


TOOL_INPUTS = {
    "create_task": CreateTask,
    "get_tasks": TaskFilter,
    "complete_task": TaskSelector,
    "update_task": UpdateTask,
    "delete_task": TaskSelector,
}


class TaskService:
    def __init__(self, repository: TaskRepository, timezone: str) -> None:
        self.repository = repository
        self.timezone = ZoneInfo(timezone)

    def _serialize(self, task: Task) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "deadline": task.deadline.astimezone(self.timezone).isoformat()
            if task.deadline
            else None,
            "priority": task.priority,
            "status": task.status,
        }

    @staticmethod
    def _invalid(error: ValidationError) -> dict[str, Any]:
        return {
            "status": "invalid_arguments",
            "errors": error.errors(include_input=False, include_context=False, include_url=False),
        }

    async def create_task(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        try:
            data = CreateTask.model_validate(arguments)
        except ValidationError as error:
            return self._invalid(error)
        reminder = next_reminder_at(data.deadline, datetime.now(UTC))
        task = await self.repository.create_task(
            telegram_user_id,
            **data.model_dump(),
            next_reminder_at=reminder,
            reminder_sent=reminder is None,
        )
        return {"status": "created", "task": self._serialize(task)}

    async def list_tasks(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        try:
            data = TaskFilter.model_validate(arguments)
        except ValidationError as error:
            return self._invalid(error)
        tasks = await self.repository.get_tasks(telegram_user_id, **data.model_dump())
        return {"status": "ok", "tasks": [self._serialize(task) for task in tasks]}

    async def get_task(self, telegram_user_id: int, task_id: int) -> dict[str, Any]:
        return await self.find_task(telegram_user_id, task_id=task_id)

    async def find_task(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        try:
            data = TaskSelector.model_validate(arguments)
        except ValidationError as error:
            return self._invalid(error)
        if data.task_id is not None:
            task = await self.repository.get_task(telegram_user_id, data.task_id)
            tasks = [task] if task else []
        else:
            tasks = await self.repository.find_tasks_by_title(telegram_user_id, data.query)
        if not tasks:
            return {"status": "not_found"}
        if len(tasks) > 1:
            return {"status": "ambiguous", "tasks": [self._serialize(task) for task in tasks]}
        return {"status": "ok", "task": self._serialize(tasks[0])}

    async def complete_task(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        result = await self.find_task(telegram_user_id, **arguments)
        if result["status"] != "ok":
            return result
        if result["task"]["status"] == "completed":
            return {"status": "already_completed", "task": result["task"]}
        task = await self.repository.complete_task(telegram_user_id, result["task"]["id"])
        return (
            {"status": "completed", "task": self._serialize(task)}
            if task
            else {"status": "not_found"}
        )

    async def update_task(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        try:
            data = UpdateTask.model_validate(arguments)
        except ValidationError as error:
            return self._invalid(error)
        selector = data.model_dump(include={"task_id", "query"}, exclude_none=True)
        result = await self.find_task(telegram_user_id, **selector)
        if result["status"] != "ok":
            return result
        changes = data.model_dump(exclude={"task_id", "query"}, exclude_unset=True)
        if "deadline" in changes:
            reminder = next_reminder_at(data.deadline, datetime.now(UTC))
            changes.update(next_reminder_at=reminder, reminder_sent=reminder is None)
        task = await self.repository.update_task(telegram_user_id, result["task"]["id"], **changes)
        return (
            {"status": "updated", "task": self._serialize(task)}
            if task
            else {"status": "not_found"}
        )

    async def delete_task(self, telegram_user_id: int, **arguments: Any) -> dict[str, Any]:
        result = await self.find_task(telegram_user_id, **arguments)
        if result["status"] != "ok":
            return result
        deleted = await self.repository.delete_task(telegram_user_id, result["task"]["id"])
        return {"status": "deleted", "task": result["task"]} if deleted else {"status": "not_found"}
