# Microduck Entertainment Robot - Web Dashboard

> **作者**: Li Changjun
> **团队**: Macao Polytechnic University - IoT Course Project 3
> **项目**: Entertainment Robot Control System

---

## 项目概述

这是一个用于控制Microduck娱乐机器人的Web Dashboard系统，支持行走模式和轮滑模式切换。

### 系统架构

```
User
  ↓
Web Dashboard (本项目)
  ↓
Cloud FastAPI Backend
  ↓
Simulation Controller
  ↓
MuJoCo Microduck Robot
```

---

## 目录结构

```
microduck-dashboard/
│
├── index.html          # 主页面
├── css/
│   └── style.css       # 样式文件
├── js/
│   ├── config.js       # 配置文件（API地址、命令等）
│   ├── api.js          # API通信层
│   ├── telemetry.js     # 遥测数据处理
│   ├── ui.js           # UI逻辑层
│   └── app.js         # 主入口
│
└── README.md           # 本文档
```

---

## 快速开始

### 1. 本地运行

```bash
# 方式一：直接用浏览器打开
open index.html

# 方式二：使用Python简易服务器
cd microduck-dashboard
python -m http.server 8080

# 方式三：使用Node.js服务器
npx serve .
```

然后访问: http://localhost:8080

### 2. 配置后端地址

编辑 `js/config.js` 文件:

```javascript
const CONFIG = {
    API_BASE_URL: "http://localhost:8000",  // 修改为实际后端地址
    API_PREFIX: "/api/v1",
    // ...
};
```

---

## 功能说明

### 模式切换

Dashboard支持两种控制模式：
- **WALK MODE (🚶)**: 行走模式 - 用于腿部行走动作
- **ROLL MODE (🛼)**: 轮滑模式 - 用于轮滑运动

点击顶部模式切换按钮可在两种模式间切换。

### 行走模式命令 (WALK MODE)

| 命令 | 说明 | 图标 |
|------|------|------|
| 前进 | 向前行走 | ⬆️ |
| 后退 | 向后行走 | ⬇️ |
| 左转 | 向左转向 | ↩️ |
| 右转 | 向右转向 | ↪️ |
| 蹲下 | 蹲下动作 | 🧎 |
| 站立 | 站立动作 | 🧍 |
| 停止 | 停止所有动作 | ⏹️ |
| 踢球(左) | 用左脚踢球 | 🦶 |
| 踢球(右) | 用右脚踢球 | 🦶 |

### 轮滑模式命令 (ROLL MODE)

| 命令 | 说明 | 图标 |
|------|------|------|
| 前进 | 向前轮滑 | ⬆️ |
| 后退 | 向后轮滑 | ⬇️ |
| 左滑 | 向左滑行 | ↩️ |
| 右滑 | 向右滑行 | ↪️ |
| 停止 | 停止滑行 | ⏹️ |

---

## API接口说明

### 发送命令

**请求**:

```http
POST /api/v1/commands
Content-Type: application/json

{
    "action": "forward"
}
```

**可用命令值**:

行走模式: `forward`, `backward`, `turn_left`, `turn_right`, `crouch`, `stand_up`, `stop`, `kick_left`, `kick_right`

轮滑模式: `roll_forward`, `roll_backward`, `roll_left`, `roll_right`, `stop`

**响应**:

```json
{
    "command_id": "xxx-xxx",
    "action": "forward",
    "status": "received"
}
```

### 获取状态

**请求**:

```http
GET /api/v1/status
```

**响应**:

```json
{
    "status": "online",
    "current_behavior": "forward",
    "current_mode": "WALK",
    "last_command": "forward",
    "timestamp": "2026-09-14T13:30:20"
}
```

### 获取遥测

**请求**:

```http
GET /api/v1/telemetry
```

**响应**:

```json
{
    "status": "online",
    "current_behavior": "forward",
    "timestamp": "2026-09-14T13:30:20",
    "joints": {
        "joint1": 0.123,
        "joint2": -0.456,
        "joint3": 0.789
    }
}
```

---

## 团队成员分工

| 成员 | 职责 |
|------|------|
| Wu Siyuan | 项目负责人，架构，系统集成，机器人仿真 |
| Zhu Jiaheng | 云后端，FastAPI，EMQX，PostgreSQL |
| **Li Changjun** | **Web Dashboard（本项目）** |
| Wang Leizhi | 遥测发布，MQTT通信 |
| Du Haofan | 测试，数据分析 |

---

## 模拟数据模式

默认启用模拟数据模式。在浏览器控制台执行:

```javascript
// 启用模拟数据
setMockMode(true);

// 禁用模拟数据（使用真实API）
setMockMode(false);
```

---

## 已知限制

1. 模拟数据模式下，遥测数据为随机生成
2. 需要后端API支持才能实现真实控制
3. 主要针对桌面/平板优化，移动端体验有限

---

**最后更新**: 2026-09-14
