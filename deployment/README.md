# Campus Agent 移动端服务器

这是一个单进程、SQLite 持久化的最小部署，适合本机、局域网或单实例内测。它不包含学生账号体系，因此共享 Bearer Token 不能作为公开多用户产品的最终认证方案。

## Docker 启动

1. 将 `.env.example` 复制为 `.env`，生成并填写至少 32 位的随机 Token。
2. 若要通过域名或局域网 IP 访问，把该主机名加入 `CAMPUS_SERVER_ALLOWED_HOSTS`。
3. 在本目录运行：

   ```powershell
   docker compose up --build -d
   ```

默认只映射到宿主机 `127.0.0.1:8000`。准备好 HTTPS 反向代理后，再把 `CAMPUS_SERVER_BIND_IP` 改为所需监听 IP；不要直接把无 TLS 的端口暴露到公网。

数据保存在 Docker volume `campus-agent-data` 中。更新容器不会删除对话 SQLite、上传知识文档和向量缓存。

## 本机启动

先安装项目并设置环境变量：

```powershell
python -m pip install -e .
$env:CAMPUS_MOBILE_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
.\start-campus-agent-server.ps1
```

服务默认监听 `127.0.0.1:8000`。若显式监听局域网地址，还要把客户端请求使用的 IP 或域名加入 `CAMPUS_SERVER_ALLOWED_HOSTS`。

## API

- `GET /api/v1/health`：公开的最小探活，不返回密钥或内部配置。
- `POST /api/v1/chat/stream`：SSE 流式聊天。
- `GET /api/v1/history?session_id=...`：读取会话历史。
- `DELETE /api/v1/history?session_id=...`：清空会话历史。

除健康检查外，请求都必须携带：

```text
Authorization: Bearer <CAMPUS_MOBILE_API_TOKEN>
```

聊天请求示例：

```powershell
curl.exe -N http://127.0.0.1:8000/api/v1/chat/stream `
  -H "Authorization: Bearer $env:CAMPUS_MOBILE_API_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{"query":"周三有什么课程？","session_id":"mobile-demo","planner":"rule"}'
```

原生 App 不需要 CORS。部署保持单个应用进程；SQLite 不适合多个容器同时写同一个数据卷。正式开放给多用户前，需要在 HTTPS 网关后增加用户登录，并由服务端根据用户身份生成或绑定 `session_id`。
