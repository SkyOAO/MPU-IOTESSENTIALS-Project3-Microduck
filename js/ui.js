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
        // 当前按钮状态
        this.buttonStates = {};
    }

    /**
     * 初始化UI
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.renderCommandButtons();
        this.updateTimestamp();
    }

    /**
     * 缓存常用DOM元素
     */
    cacheElements() {
        this.elements = {
            // 状态卡片
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            currentBehavior: document.getElementById('current-behavior'),
            lastUpdate: document.getElementById('last-update'),
            // 反馈消息
            feedbackMessage: document.getElementById('feedback-message'),
            // 命令按钮容器
            commandButtons: document.getElementById('command-buttons')
        };
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 遥测数据更新时更新状态
        telemetryService.onUpdate((data) => {
            this.updateFromTelemetry(data);
        });

        // 更新时间戳
        setInterval(() => this.updateTimestamp(), 1000);
    }

    /**
     * 从遥测数据更新UI
     * @param {Object} data - 遥测数据
     */
    updateFromTelemetry(data) {
        if (!data) return;

        // 更新连接状态
        if (data.status) {
            this.updateConnectionStatus(data.status);
        }

        // 更新当前行为
        if (data.current_behavior) {
            this.updateCurrentBehavior(data.current_behavior);
        }
    }

    /**
     * 渲染命令按钮（7个按钮一行排列）
     */
    renderCommandButtons() {
        if (!this.elements.commandButtons) return;

        const container = this.elements.commandButtons;
        container.innerHTML = '';

        COMMANDS.forEach(cmd => {
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
            this.updateCurrentBehavior(action.toUpperCase());
        } else {
            this.showFeedback('error', `✕ Failed: ${result.error}`);
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

        if (!feedback) return;

        feedback.textContent = message;
        feedback.className = `feedback-message ${type}`;
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
            text.textContent = config.text;
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
     * 更新时间戳显示
     */
    updateTimestamp() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();

        if (this.elements.lastUpdate) {
            this.elements.lastUpdate.textContent = timeString;
        }
    }
}

// 创建全局UI管理器实例
const uiManager = new UIManager();

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UIManager, uiManager };
}
