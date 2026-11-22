FROM python:3.13-slim
WORKDIR /bot
COPY bot.py brain.py world.py mechio.py model.py /bot/
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
