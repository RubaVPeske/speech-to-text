import os
import tempfile
import asyncio
import warnings
from dotenv import load_dotenv
from aiogram import Dispatcher, F, types
from aiogram.client.bot import Bot, DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode
from aiogram.filters import Command
import whisper
import g4f
import torch

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)


async def transcribe_audio(file_path: str) -> str:
    result = model.transcribe(file_path)
    return result["text"].strip()


async def generate_summary(text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the user text:"},
        {"role": "user", "content": text},
    ]
    try:
        response = g4f.ChatCompletion.create(model="gpt-4o-mini", messages=messages, stream=False)
        if response:
            return response
        else:
            return "Ошибка: пустой ответ от g4f."
    except Exception as e:
        return f"Не удалось получить ответ от g4f: {str(e)}"


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name or ""
    text = f"Привет, {user_name}!\nЯ бот, который умеет:\n1. Принимать голосовые сообщения (ogg)\n2. Транскрибировать их (Whisper)\n3. С помощью g4f (ChatGPT) – генерировать краткое резюме.\n\nОтправьте голосовое или текст, а я верну ответ.\n\n(Whisper работает на {device})\nБот доступен для всех пользователей!"
    await message.answer(text)


@dp.message(F.voice)
async def handle_voice(message: types.Message):
    status_msg = await message.answer("Обрабатываю голосовое...")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            temp_path = temp_file.name
        await bot.download_file(file_info.file_path, destination=temp_path)
        transcription = await transcribe_audio(temp_path)
        # summary = await generate_summary(transcription)
        if len(transcription) > 3000:
            await status_msg.edit_text("Транскрибация (разбита на части)")
            chunk_size = 4000
            transcription_chunks = [transcription[i:i + chunk_size] for i in range(0, len(transcription), chunk_size)]
            for idx, chunk in enumerate(transcription_chunks, 1):
                await message.answer(f"Часть {idx}:\n{chunk}")
            # await message.answer(f"Краткое резюме:\n{summary}") ffmpeg -version

        else:
            full_text = f"Транскрибация:\n{transcription}" # \n\nКраткое резюме:\n{summary}
            await status_msg.edit_text(full_text)
        os.unlink(temp_path)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")


@dp.message(F.text)
async def handle_text(message: types.Message):
    status_msg = await message.answer("Генерирую краткое резюме...")
    try:
        text_input = message.text
        summary = await generate_summary(text_input)
        if len(summary) > 4000:
            chunk_size = 4000
            summary_chunks = [summary[i:i + chunk_size] for i in range(0, len(summary), chunk_size)]
            await status_msg.edit_text("Результат слишком большой, отправляю частями:")
            for idx, chunk in enumerate(summary_chunks, 1):
                await message.answer(f"Часть {idx}:\n{chunk}")
        else:
            await status_msg.edit_text(f"Краткое резюме:\n{summary}")
    except Exception as e:
        await message.answer(f"Ошибка при генерации: {e}")


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
