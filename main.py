import asyncio
import logging
from datetime import UTC, datetime

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramForbiddenError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from groq import AsyncGroq

from bot import create_router
from config import Settings
from database import Database
from llm_service import LLMService, format_deadline
from repository import TaskRepository
from task_service import TaskService, next_reminder_at

logger = logging.getLogger(__name__)


async def send_reminders(bot: Bot, repository: TaskRepository) -> None:
    for telegram_id, task in await repository.get_pending_reminders(datetime.now(UTC)):
        now = datetime.now(UTC)
        if task.deadline <= now:
            await repository.advance_reminder(
                telegram_id, task.id, task.deadline, task.next_reminder_at, None
            )
            continue
        try:
            minutes = max(1, int((task.deadline - now).total_seconds() / 60))
            days, minutes = divmod(minutes, 1440)
            hours, minutes = divmod(minutes, 60)
            remaining = ", ".join(
                f"{value} {unit}"
                for value, unit in ((days, "дн."), (hours, "ч."), (minutes, "мин."))
                if value
            )
            await bot.send_message(
                telegram_id,
                f"⏰ Напоминание: {task.title}\n"
                f"Срок: {format_deadline(task.deadline.isoformat())}\n"
                f"До срока: {remaining}",
                parse_mode=None,
            )
        except TelegramForbiddenError:
            await repository.advance_reminder(
                telegram_id, task.id, task.deadline, task.next_reminder_at, None
            )
            logger.info("Reminder skipped: bot is blocked")
        except Exception as error:
            logger.warning("Reminder delivery failed: %s", type(error).__name__)
        else:
            await repository.advance_reminder(
                telegram_id,
                task.id,
                task.deadline,
                task.next_reminder_at,
                next_reminder_at(task.deadline, datetime.now(UTC)),
            )


async def initialize_reminders(repository: TaskRepository) -> None:
    while tasks := await repository.get_unscheduled_reminders():
        for telegram_id, task in tasks:
            await repository.advance_reminder(
                telegram_id,
                task.id,
                task.deadline,
                None,
                next_reminder_at(task.deadline, datetime.now(UTC)),
            )


async def main() -> None:
    settings = Settings()
    database = Database(settings.database_url.get_secret_value())
    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    bot: Bot | None = None
    client: AsyncGroq | None = None
    reminder_job: asyncio.Task | None = None
    try:
        await database.initialize()
        repository = TaskRepository(database.sessions)
        await initialize_reminders(repository)
        task_service = TaskService(repository, settings.timezone)
        client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(), timeout=30.0, max_retries=0
        )
        llm_service = LLMService(client, task_service, settings.groq_model, settings.timezone)
        bot = Bot(token=settings.telegram_bot_token.get_secret_value())
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router(task_service, llm_service))

        async def run_reminders() -> None:
            nonlocal reminder_job
            reminder_job = asyncio.current_task()
            await send_reminders(bot, repository)

        scheduler.add_job(
            run_reminders,
            "interval",
            minutes=settings.reminder_check_minutes,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(UTC),
        )
        scheduler.start()
        logger.info("Database initialized; reminders and polling starting")
        await dispatcher.start_polling(bot, close_bot_session=False, allowed_updates=["message"])
    finally:
        if scheduler.running:
            scheduler.pause()
            await asyncio.sleep(0)
            if reminder_job is not None:
                await asyncio.gather(reminder_job, return_exceptions=True)
            scheduler.shutdown(wait=False)
            await asyncio.sleep(0)
        if bot is not None:
            await bot.session.close()
        if client is not None:
            await client.close()
        await database.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
