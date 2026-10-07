/**
 * Microduck Dashboard 配置
 *
 * 作者: Li Changjun
 * 团队: Macao Polytechnic University - IoT Course Project 3
 */

// ============================================
// 重要: 修改此配置以连接到实际后端
// ============================================

// API 配置
const CONFIG = {
    // 后端服务器地址 - 修改为实际服务器地址
    API_BASE_URL: "http://8.138.114.112:8000",

    // API 版本前缀
    API_PREFIX: "/api/v1",

    // 请求超时时间(毫秒)
    REQUEST_TIMEOUT: 10000,

    // 遥测轮询间隔(毫秒) - 设置为false可禁用轮询
    TELEMETRY_POLL_INTERVAL: 2000,
};

// ============================================
// 命令配置
// ============================================

// 命令名称映射 (前端显示名 -> 后端接受的值)
const COMMAND_MAP = {
    "FORWARD": "forward",
    "BACKWARD": "backward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "CROUCH_STAND": "crouch_stand",
    "DANCE": "dance",
    "STOP": "stop"
};

// 命令按钮配置（7个按钮：前进、后退、左转、右转、蹲下/起立、跳舞、停止）
const COMMANDS = [
    { id: "forward", label: "前进", icon: "⬆️" },
    { id: "backward", label: "后退", icon: "⬇️" },
    { id: "turn_left", label: "左转", icon: "↩️" },
    { id: "turn_right", label: "右转", icon: "↪️" },
    { id: "crouch_stand", label: "蹲下/起立", icon: "🧎<span class='slash'>/</span>🧍" },
    { id: "dance", label: "跳舞", icon: "💃" },
    { id: "stop", label: "停止", icon: "⏹️" },
];

// ============================================
// 状态配置
// ============================================

const STATUS_CONFIG = {
    online: {
        text: "ONLINE",
        color: "#22c55e"
    },
    offline: {
        text: "OFFLINE",
        color: "#ef4444"
    }
};

// ============================================
// 遥测字段配置
// ============================================

const TELEMETRY_FIELDS = [
    { key: 'joint1', label: 'Joint 1', unit: 'rad', precision: 3 },
    { key: 'joint2', label: 'Joint 2', unit: 'rad', precision: 3 },
    { key: 'joint3', label: 'Joint 3', unit: 'rad', precision: 3 },
    { key: 'joint4', label: 'Joint 4', unit: 'rad', precision: 3 },
    { key: 'joint5', label: 'Joint 5', unit: 'rad', precision: 3 },
    { key: 'joint6', label: 'Joint 6', unit: 'rad', precision: 3 },
];

// ============================================
// API 端点配置
// ============================================

const API_ENDPOINTS = {
    commands: `${CONFIG.API_BASE_URL}${CONFIG.API_PREFIX}/commands`,
    status: `${CONFIG.API_BASE_URL}${CONFIG.API_PREFIX}/status`,
    telemetry: `${CONFIG.API_BASE_URL}${CONFIG.API_PREFIX}/telemetry`
};

// ============================================
// 导出配置供其他模块使用
// ============================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, COMMAND_MAP, COMMANDS, STATUS_CONFIG, TELEMETRY_FIELDS, API_ENDPOINTS };
}
