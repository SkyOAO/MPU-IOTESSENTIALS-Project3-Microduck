/**
 * Microduck Dashboard - 遥测数据处理
 *
 * 作者: Li Changjun
 * 团队: Macao Polytechnic University - IoT Course Project 3
 */

class TelemetryService {
    constructor() {
        this.data = null;
        this.lastUpdate = null;
        this.pollingInterval = null;
        this.callbacks = [];
        this.useMockData = true; // 默认使用模拟数据
        this.mockInterval = null;
    }

    /**
     * 启动遥测数据轮询
     * @param {number} interval - 轮询间隔(毫秒)
     */
    startPolling(interval = CONFIG.TELEMETRY_POLL_INTERVAL) {
        if (!interval) {
            console.log('Telemetry polling disabled');
            return;
        }

        this.stopPolling();

        // 立即获取一次数据
        this.fetchTelemetry();

        // 设置定期轮询
        this.pollingInterval = setInterval(() => {
            this.fetchTelemetry();
        }, interval);

        console.log(`Telemetry polling started (interval: ${interval}ms)`);
    }

    /**
     * 停止遥测数据轮询
     */
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    /**
     * 获取遥测数据
     */
    async fetchTelemetry() {
        if (this.useMockData) {
            // 使用模拟数据
            this.updateMockData();
            return;
        }

        try {
            const result = await apiService.getTelemetry();

            if (result.success) {
                this.data = result.data;
                this.lastUpdate = new Date();
                this.notifyCallbacks(result.data);
            } else {
                console.warn('Telemetry fetch failed:', result.error);
            }
        } catch (error) {
            console.error('Telemetry error:', error);
        }
    }

    /**
     * 注册数据更新回调
     * @param {Function} callback - 回调函数
     */
    onUpdate(callback) {
        if (typeof callback === 'function') {
            this.callbacks.push(callback);
        }
    }

    /**
     * 移除回调
     * @param {Function} callback - 回调函数
     */
    offUpdate(callback) {
        this.callbacks = this.callbacks.filter(cb => cb !== callback);
    }

    /**
     * 通知所有回调
     * @param {Object} data - 遥测数据
     */
    notifyCallbacks(data) {
        this.callbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Telemetry callback error:', error);
            }
        });
    }

    /**
     * 更新模拟数据
     */
    updateMockData() {
        const mockTelemetry = {
            status: this.getMockStatus(),
            current_behavior: this.getMockBehavior(),
            last_command: this.getMockLastCommand(),
            timestamp: new Date().toISOString(),
            joints: {
                joint1: this.generateRandomJoint(),
                joint2: this.generateRandomJoint(),
                joint3: this.generateRandomJoint(),
                joint4: this.generateRandomJoint(),
                joint5: this.generateRandomJoint(),
                joint6: this.generateRandomJoint()
            }
        };

        this.data = mockTelemetry;
        this.lastUpdate = new Date();
        this.notifyCallbacks(mockTelemetry);
    }

    /**
     * 生成随机关节值
     */
    generateRandomJoint() {
        return (Math.random() * 2 - 1).toFixed(3);
    }

    /**
     * 获取模拟状态
     */
    getMockStatus() {
        return 'online';
    }

    /**
     * 获取模拟行为
     */
    getMockBehavior() {
        const behaviors = ['idle', 'forward', 'backward', 'turn_left', 'turn_right',
                         'crouch', 'stand_up', 'stop', 'kick_left', 'kick_right',
                         'roll_forward', 'roll_backward', 'roll_left', 'roll_right'];
        return behaviors[Math.floor(Math.random() * behaviors.length)];
    }

    /**
     * 获取模拟最后命令
     */
    getMockLastCommand() {
        const commands = ['forward', 'backward', 'turn_left', 'turn_right',
                         'crouch', 'stand_up', 'stop', 'kick_left', 'kick_right',
                         'roll_forward', 'roll_backward', 'roll_left', 'roll_right'];
        return commands[Math.floor(Math.random() * commands.length)];
    }

    /**
     * 获取当前数据
     */
    getData() {
        return this.data;
    }

    /**
     * 获取最后更新时间
     */
    getLastUpdate() {
        return this.lastUpdate;
    }

    /**
     * 启用/禁用模拟数据
     * @param {boolean} enabled
     */
    setMockDataEnabled(enabled) {
        this.useMockData = enabled;
        if (enabled) {
            this.stopPolling();
            this.updateMockData();
        } else {
            this.startPolling(CONFIG.TELEMETRY_POLL_INTERVAL);
        }
    }
}

// 创建全局遥测服务实例
const telemetryService = new TelemetryService();

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TelemetryService, telemetryService };
}
