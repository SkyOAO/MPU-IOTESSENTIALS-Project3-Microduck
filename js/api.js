/**
 * Microduck Dashboard - API 服务层
 *
 * 作者: Li Changjun
 * 团队: Macao Polytechnic University - IoT Course Project 3
 */

class ApiService {
    constructor() {
        this.baseUrl = CONFIG.API_BASE_URL;
        this.prefix = CONFIG.API_PREFIX;
        this.timeout = CONFIG.REQUEST_TIMEOUT;
    }

    /**
     * 获取完整的API URL
     */
    getEndpoint(endpoint) {
        return `${this.baseUrl}${this.prefix}${endpoint}`;
    }

    /**
     * 发送命令到后端
     * @param {string} action - 命令动作 (forward, backward, turn_left, 等)
     * @returns {Promise} 返回命令结果
     */
    async sendCommand(action) {
        const url = this.getEndpoint('/commands');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            // 构造符合后端API格式的请求体
            const requestBody = {
                op: action,
                params: {},
                robot_id: "robot01",
                msg_id: `msg_${Date.now()}`,
                timestamp: Math.floor(Date.now() / 1000)
            };

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': 'e373e7b39ee0174c1fefeb5cf2a66bc2',
                },
                body: JSON.stringify(requestBody),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.text();
            return {
                success: true,
                data: data,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            if (error.name === 'AbortError') {
                return {
                    success: false,
                    error: 'Request timed out',
                    timestamp: new Date().toISOString()
                };
            }
            return {
                success: false,
                error: error.message || 'Failed to send command',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * 获取机器人状态
     * @returns {Promise} 返回状态数据
     */
    async getStatus() {
        const url = this.getEndpoint('/status');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'X-API-Key': 'e373e7b39ee0174c1fefeb5cf2a66bc2',
                },
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            if (error.name === 'AbortError') {
                return {
                    success: false,
                    error: 'Request timed out',
                    timestamp: new Date().toISOString()
                };
            }
            return {
                success: false,
                error: error.message || 'Failed to get status',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * 获取遥测数据
     * @returns {Promise} 返回遥测数据
     */
    async getTelemetry() {
        const url = this.getEndpoint('/telemetry');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'X-API-Key': 'e373e7b39ee0174c1fefeb5cf2a66bc2',
                },
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            if (error.name === 'AbortError') {
                return {
                    success: false,
                    error: 'Request timed out',
                    timestamp: new Date().toISOString()
                };
            }
            return {
                success: false,
                error: error.message || 'Failed to get telemetry',
                timestamp: new Date().toISOString()
            };
        }
    }
}

// 创建全局API服务实例
const apiService = new ApiService();

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ApiService, apiService };
}
