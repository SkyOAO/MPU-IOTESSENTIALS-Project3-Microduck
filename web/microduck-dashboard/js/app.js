/**
 * Microduck Dashboard - 主入口
 *
 * 作者: Li Changjun
 * 团队: Macao Polytechnic University - IoT Course Project 3
 */

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('Microduck Dashboard initializing...');

    // 初始化UI
    uiManager.init();

    // 启动遥测数据轮询（使用模拟数据）
    telemetryService.startPolling(2000);

    // 初始状态
    uiManager.updateConnectionStatus('online');

    console.log('Microduck Dashboard ready!');
});

/**
 * 切换模拟数据模式
 * @param {boolean} enabled - 是否启用模拟数据
 */
function setMockMode(enabled) {
    telemetryService.setMockDataEnabled(enabled);
    console.log(`Mock mode: ${enabled ? 'enabled' : 'disabled'}`);
}

/**
 * 手动刷新遥测数据
 */
function refreshTelemetry() {
    telemetryService.fetchTelemetry();
}

/**
 * 获取API服务实例
 */
function getApiService() {
    return apiService;
}

/**
 * 获取遥测服务实例
 */
function getTelemetryService() {
    return telemetryService;
}

/**
 * 获取UI管理器实例
 */
function getUIManager() {
    return uiManager;
}
