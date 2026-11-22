FROM python:3.13-slim
WORKDIR /bot
COPY bot.py mechio.py /bot/
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
