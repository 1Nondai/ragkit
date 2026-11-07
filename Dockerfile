# 应用镜像（只装运行时依赖，dev 组不进镜像）
FROM python:3.12-slim

# PYTHONUNBUFFERED=1：日志立刻输出，否则容器日志卡在缓冲区，docker logs 看不到实时输出
# PYTHONDONTWRITEBYTECODE=1：不生成 .pyc，镜像里少一层垃圾
# PIP_NO_CACHE_DIR=1：不保留 pip 缓存，镜像小几十 MB
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先拷依赖清单，利用 Docker 层缓存
COPY pyproject.toml README.md ./
COPY src/ ./src/

# 关键：必须装 [runtime] 这组额外依赖
# 只写 `pip install -e .` 只会装 pydantic，uvicorn / fastapi / pymilvus 全都缺，
# 容器启动直接 ModuleNotFoundError: No module named 'uvicorn'
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[runtime]"

# 语料与评测集按需挂载，不打进镜像
VOLUME ["/app/data"]

EXPOSE 8099

# 生产请用进程管理器或编排系统控制 worker 数量
CMD ["uvicorn", "ragkit.api.server:app", "--host", "0.0.0.0", "--port", "8099"]
