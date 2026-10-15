import logging

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender

from llm_service import LLMService, format_deadline
from task_service import TaskService

logger = logging.getLogger(__name__)
HELP_TEXT = """Привет! Я AI Task Manager. Пиши задачи обычным языком:

• Завтра в 18:00 сделать лабораторную
• Что у меня завтра?
• Я закончил лабораторную
• Перенеси лабораторную на пятницу в 15:00
• Удали задачу купить продукты

/tasks — активные задачи без обращения к ИИ
/help — эта справка

Если задачи похожи, укажи название точнее или ID из /tasks.
Даты в ответах показаны по московскому времени (МСК)."""


async def send_text(message: Message, text: str) -> None:
    for start in range(0, len(text), 3500):
        await message.answer(text[start : start + 3500], parse_mode=None)


def create_router(task_service: TaskService, llm_service: LLMService) -> Router:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(Command("tasks"))
    async def tasks_handler(message: Message) -> None:
        if message.from_user is None:
            return
        result = await task_service.list_tasks(message.from_user.id)
        if not result["tasks"]:
            await message.answer("Активных задач пока нет. Напиши, что нужно сделать.")
            return
        priorities = {"low": "низкий", "medium": "средний", "high": "высокий"}
        lines = ["Активные задачи (МСК):"]
        for task in result["tasks"]:
            deadline = format_deadline(task["deadline"]) if task["deadline"] else "без срока"
            lines.append(
                f"#{task['id']} — {task['title']}\n{deadline}, приоритет: {priorities[task['priority']]}"
            )
        await send_text(message, "\n\n".join(lines))

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message) -> None:
        await message.answer("Доступны /start, /help и /tasks. Или напиши задачу обычным языком.")

    @router.message(F.text)
    async def text_handler(message: Message) -> None:
        if message.from_user is None or message.text is None:
            return
        await task_service.repository.get_or_create_user(
            message.from_user.id, message.from_user.username
        )
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            answer = await llm_service.respond(message.from_user.id, message.text)
        await send_text(message, answer)

    @router.message()
    async def unsupported_handler(message: Message) -> None:
        await message.answer("Пока я понимаю только текст. Напиши задачу сообщением.")

    @router.errors()
    async def error_handler(event: ErrorEvent) -> bool:
        logger.error("Telegram handler failed: %s", type(event.exception).__name__)
        if event.update.message:
            try:
                await event.update.message.answer(
                    "Не удалось обработать сообщение. Попробуйте позже."
                )
            except Exception as error:
                logger.warning("Could not send error message: %s", type(error).__name__)
        return True

    return router
