# Campus Agent Mobile

最小 Expo/React Native 客户端，调用项目根目录服务器的 `/api/v1` 接口。App 只保存服务器地址、共享访问 Token、规划器偏好和随机会话 ID；Agent、知识库、SQLite、DeepSeek Key 都留在服务器。

## 运行

本机没有 Android SDK 时，推荐在同一 Wi-Fi 下使用 Expo Go：

1. 查询电脑局域网 IPv4，例如 `192.168.1.8`。
2. 在项目根目录启动服务器：

   ```powershell
   $mobileToken = python -c "import secrets; print(secrets.token_urlsafe(32))"
   $env:CAMPUS_MOBILE_API_TOKEN = $mobileToken
   $mobileToken  # 复制这个值，稍后填入 App
   $env:CAMPUS_SERVER_HOST = "0.0.0.0"
   $env:CAMPUS_SERVER_ALLOWED_HOSTS = "192.168.1.8"
   .\start-campus-agent-server.ps1
   ```

3. 新开终端并启动 App：

   ```powershell
   .\start-campus-agent-mobile.ps1
   ```

4. 在手机安装 Expo Go，扫描终端二维码。
5. 首次打开后填写：
   - 服务器：`http://192.168.1.8:8000`
   - Token：步骤 2 生成的 `CAMPUS_MOBILE_API_TOKEN`

手机不能使用 `127.0.0.1` 访问电脑。若 Windows 防火墙弹窗，需要允许 Python 和 Node 在“专用网络”通信。

Android 模拟器访问宿主机时可使用 `http://10.0.2.2:8000`。正式公网部署必须使用 HTTPS。

## 验证命令

```powershell
cd mobile-app
npm.cmd run typecheck
npm.cmd test
npm.cmd run export:android
```

## 当前边界

- 支持 Android、iOS 的 Expo Go/Expo 构建；未包含商店签名和发布流程。
- 支持 SSE 进度与最终回答、服务端历史、清空会话、Rule/DeepSeek 切换。
- Token 使用 Android Keystore/iOS Keychain 对应的 Expo SecureStore。
- 共享 Token 适合内测，不提供真实学生账号隔离；正式发布前应由服务器签发用户级 Token，并绑定用户与会话。
