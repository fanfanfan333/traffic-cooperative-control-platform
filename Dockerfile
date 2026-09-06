# 车路云一体化协同管控仿真平台 —— 容器化部署
# 功能二③"平台部署": 构建: docker build -t traffic-cloud .  运行: docker run -p 8080:8080 traffic-cloud
# 容器内默认启动实时数据可视化服务器(云端), 数据目录挂载到宿主机即可看板实时刷新:
#   docker run -p 8080:8080 -v $(pwd)/collected_data:/app/collected_data traffic-cloud
FROM python:3.11-slim

# SUMO 仿真内核(Ubuntu官方源) + 环境变量
RUN apt-get update && apt-get install -y --no-install-recommends sumo sumo-tools \
    && rm -rf /var/lib/apt/lists/*
ENV SUMO_HOME=/usr/share/sumo

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "dashboard_server.py", "--host", "0.0.0.0", "--port", "8080"]
