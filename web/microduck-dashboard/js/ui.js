/**
 * Microduck Dashboard - UI 逻辑层
 *
 * 作者: Li Changjun
 * 团队: Macao Polytechnic University - IoT Course Project 3
 */

class UIManager {
    constructor() {
        // DOM元素缓存
        this.elements = {};
        // 命令历史
        this.commandHistory = [];
        // 最大历史记录数
        this.maxHistorySize = 10;
        // 当前按钮状态
        this.buttonStates = {};
    }

    /**
     * 初始化UI
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.renderModeToggle();
        this.renderCommandButtons();
        this.renderTelemetryFields();
        this.updateTimestamp();
        this.updateModeDisplay();
    }

    /**
     * 缓存常用DOM元素
     */
    cacheElements() {
        this.elements = {
            // 状态卡片
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            connectionStatus: document.getElementById('connection-status'),
            currentBehavior: document.getElementById('current-behavior'),
            lastCommand: document.getElementById('last-command'),
            lastUpdate: document.getElementById('last-update'),

            // 遥测面板
            telemetryBody: document.getElementById('telemetry-body'),
            telemetryStatus: document.getElementById('telemetry-status'),
            telemetryUpdate: document.getElementById('telemetry-update'),

            // 命令历史
            historyBody: document.getElementById('history-body'),
            historyCount: document.getElementById('history-count'),

            // 反馈消息
            feedbackMessage: document.getElementById('feedback-message'),
            feedbackType: document.getElementById('feedback-type'),

            // 命令按钮容器
            commandButtons: document.getElementById('command-buttons'),

            // 模式切换容器
            modeToggle: document.getElementById('mode-toggle'),
            currentModeDisplay: document.getElementById('current-mode')
        };
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 遥测数据更新
        telemetryService.onUpdate((data) => {
            this.updateTelemetry(data);
        });

        // 更新时间戳
        setInterval(() => this.updateTimestamp(), 1000);
    }

    /**
     * 渲染模式切换按钮
     */
    renderModeToggle() {
        if (!this.elements.modeToggle) return;

        const container = this.elements.modeToggle;
        container.innerHTML = `
            <button id="mode-btn" class="mode-toggle-btn">
                <span class="mode-icon">${MODE_CONFIG[CURRENT_MODE].icon}</span>
                <span class="mode-label">${MODE_CONFIG[CURRENT_MODE].name}</span>
            </button>
        `;

        document.getElementById('mode-btn').addEventListener('click', () => {
            this.toggleMode();
        });
    }

    /**
     * 切换模式
     */
    toggleMode() {
        CURRENT_MODE = CURRENT_MODE === "WALK" ? "ROLL" : "WALK";
        this.updateModeDisplay();
        this.renderCommandButtons();
    }

    /**
     * 更新模式显示
     */
    updateModeDisplay() {
        const btn = document.getElementById('mode-btn');
        if (btn) {
            btn.innerHTML = `
                <span class="mode-icon">${MODE_CONFIG[CURRENT_MODE].icon}</span>
                <span class="mode-label">${MODE_CONFIG[CURRENT_MODE].name}</span>
            `;
            btn.style.borderColor = MODE_CONFIG[CURRENT_MODE].color;
        }

        if (this.elements.currentModeDisplay) {
            this.elements.currentModeDisplay.textContent = MODE_CONFIG[CURRENT_MODE].name;
            this.elements.currentModeDisplay.style.color = MODE_CONFIG[CURRENT_MODE].color;
        }
    }

    /**
     * 获取当前模式的命令列表
     */
    getCurrentCommands() {
        return CURRENT_MODE === "WALK" ? WALK_COMMANDS : ROLL_COMMANDS;
    }

    /**
     * 渲染命令按钮
     */
    renderCommandButtons() {
        if (!this.elements.commandButtons) return;

        const container = this.elements.commandButtons;
        container.innerHTML = '';

        const commands = this.getCurrentCommands();
        commands.forEach(cmd => {
            const button = document.createElement('button');
            button.id = `btn-${cmd.id}`;
            button.className = 'command-btn';
            button.dataset.command = cmd.id;
            button.innerHTML = `
                <span class="cmd-icon">${cmd.icon}</span>
                <span class="cmd-label">${cmd.label}</span>
            `;
            button.addEventListener('click', () => this.handleCommandClick(cmd.id));
            container.appendChild(button);
        });
    }

    /**
     * 渲染遥测字段
     */
    renderTelemetryFields() {
        if (!this.elements.telemetryBody) return;

        const container = this.elements.telemetryBody;
        container.innerHTML = '';

        TELEMETRY_FIELDS.forEach(field => {
            const row = document.createElement('tr');
            row.id = `telemetry-${field.key}`;
            row.innerHTML = `
                <td class="field-label">${field.label}</td>
                <td class="field-value" id="${field.key}-value">--</td>
                <td class="field-unit">${field.unit}</td>
            `;
            container.appendChild(row);
        });
    }

    /**
     * 处理命令按钮点击
     * @param {string} commandId - 命令ID
     */
    async handleCommandClick(commandId) {
        // 禁用按钮
        this.setButtonLoading(commandId, true);

        // 显示发送中状态
        this.showFeedback('sending', `Sending ${commandId.toUpperCase()}...`);

        // 获取后端接受的动作名称
        const action = COMMAND_MAP[commandId.toUpperCase()] || commandId;

        // 发送到后端
        const result = await apiService.sendCommand(action);

        if (result.success) {
            this.showFeedback('success', `✓ Command "${action}" sent successfully`);
            this.addToHistory(action, 'SUCCESS');
            this.updateLastCommand(action.toUpperCase());
        } else {
            this.showFeedback('error', `✕ Failed: ${result.error}`);
            this.addToHistory(action, 'FAILED');
        }

        // 恢复按钮
        setTimeout(() => {
            this.setButtonLoading(commandId, false);
        }, 500);

        // 3秒后清除反馈
        setTimeout(() => {
            this.clearFeedback();
        }, 3000);
    }

    /**
     * 设置按钮加载状态
     * @param {string} commandId - 命令ID
     * @param {boolean} loading - 是否加载中
     */
    setButtonLoading(commandId, loading) {
        const button = document.getElementById(`btn-${commandId}`);
        if (!button) return;

        if (loading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    }

    /**
     * 显示反馈消息
     * @param {string} type - 消息类型 (success, error, sending)
     * @param {string} message - 消息内容
     */
    showFeedback(type, message) {
        const feedback = this.elements.feedbackMessage;
        const feedbackType = this.elements.feedbackType;

        if (!feedback || !feedbackType) return;

        feedback.textContent = message;
        feedback.className = `feedback-message ${type}`;
        feedbackType.textContent = type === 'success' ? '✓' : type === 'error' ? '✕' : '⏳';
    }

    /**
     * 清除反馈消息
     */
    clearFeedback() {
        const feedback = this.elements.feedbackMessage;
        if (feedback) {
            feedback.textContent = '';
            feedback.className = 'feedback-message';
        }
    }

    /**
     * 添加命令到历史记录
     * @param {string} command - 命令
     * @param {string} status - 状态
     */
    addToHistory(command, status) {
        const timestamp = new Date().toLocaleTimeString();
        const record = { command, status, timestamp };

        this.commandHistory.unshift(record);

        // 限制历史长度
        if (this.commandHistory.length > this.maxHistorySize) {
            this.commandHistory.pop();
        }

        this.renderHistory();
    }

    /**
     * 渲染命令历史
     */
    renderHistory() {
        const container = this.elements.historyBody;
        const countEl = this.elements.historyCount;

        if (!container) return;

        if (this.commandHistory.length === 0) {
            container.innerHTML = '<tr><td colspan="3" class="empty-history">No commands yet</td></tr>';
        } else {
            container.innerHTML = this.commandHistory.map(record => `
                <tr class="history-row ${record.status.toLowerCase()}">
                    <td class="history-time">${record.timestamp}</td>
                    <td class="history-command">${record.command.toUpperCase()}</td>
                    <td class="history-status ${record.status.toLowerCase()}">${record.status}</td>
                </tr>
            `).join('');
        }

        if (countEl) {
            countEl.textContent = this.commandHistory.length;
        }
    }

    /**
     * 更新连接状态
     * @param {string} status - 状态 (online, offline, connecting, error)
     */
    updateConnectionStatus(status) {
        const indicator = this.elements.statusIndicator;
        const text = this.elements.statusText;
        const config = STATUS_CONFIG[status] || STATUS_CONFIG.offline;

        if (indicator) {
            indicator.className = `status-dot ${status}`;
        }

        if (text) {
            text.textContent = `${config.icon} ${config.text}`;
            text.style.color = config.color;
        }
    }

    /**
     * 更新当前行为显示
     * @param {string} behavior - 行为名称
     */
    updateCurrentBehavior(behavior) {
        if (this.elements.currentBehavior) {
            this.elements.currentBehavior.textContent = behavior ? behavior.toUpperCase() : '--';
        }
    }

    /**
     * 更新最后命令显示
     * @param {string} command - 命令
     */
    updateLastCommand(command) {
        if (this.elements.lastCommand) {
            this.elements.lastCommand.textContent = command || '--';
        }
    }

    /**
     * 更新时间戳显示
     */
    updateTimestamp() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();

        if (this.elements.lastUpdate) {
            this.elements.lastUpdate.textContent = timeString;
        }

        if (this.elements.telemetryUpdate) {
            this.elements.telemetryUpdate.textContent = timeString;
        }
    }

    /**
     * 更新遥测数据
     * @param {Object} data - 遥测数据
     */
    updateTelemetry(data) {
        if (!data) {
            this.showTelemetryError();
            return;
        }

        // 更新状态信息
        if (data.status) {
            this.updateConnectionStatus(data.status);
        }

        if (data.current_behavior) {
            this.updateCurrentBehavior(data.current_behavior);
        }

        // 更新关节数据
        if (data.joints) {
            TELEMETRY_FIELDS.forEach(field => {
                const valueEl = document.getElementById(`${field.key}-value`);
                if (valueEl && data.joints[field.key] !== undefined) {
                    const value = parseFloat(data.joints[field.key]).toFixed(field.precision);
                    valueEl.textContent = value;
                }
            });
        }

        // 更新时间
        if (data.timestamp) {
            const time = new Date(data.timestamp).toLocaleTimeString();
            if (this.elements.telemetryUpdate) {
                this.elements.telemetryUpdate.textContent = time;
            }
        }

        // 显示遥测面板
        this.hideTelemetryError();
    }

    /**
     * 显示遥测错误
     */
    showTelemetryError() {
        if (this.elements.telemetryStatus) {
            this.elements.telemetryStatus.textContent = 'UNAVAILABLE';
            this.elements.telemetryStatus.className = 'telemetry-status error';
        }
    }

    /**
     * 隐藏遥测错误
     */
    hideTelemetryError() {
        if (this.elements.telemetryStatus) {
            this.elements.telemetryStatus.textContent = 'NORMAL';
            this.elements.telemetryStatus.className = 'telemetry-status';
        }
    }

    /**
     * 显示后端连接错误
     */
    showBackendError(message) {
        this.updateConnectionStatus('error');
        this.showFeedback('error', `Backend Error: ${message}`);
    }
}

// 创建全局UI管理器实例
const uiManager = new UIManager();

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UIManager, uiManager };
}
