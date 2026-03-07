"""Training logger for RL algorithms.

Provides a unified interface for logging training metrics to TensorBoard.
"""
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Handles logging of training metrics to TensorBoard.

    Extracts logging responsibility from algorithm classes to provide
    a reusable, testable logging interface.
    """

    def __init__(self, log_dir, flush_secs: int = 10):
        """Initialize the training logger.

        Args:
            log_dir: Directory to save TensorBoard logs.
            flush_secs: How often to flush logs to disk.
        """
        self.log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Args:
            tag: Name of the metric (e.g., "Loss/actor").
            value: Scalar value to log.
            step: Training step/iteration.
        """
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values, step: int) -> None:
        """Log multiple related scalars.

        Args:
            main_tag: Group name for the metrics.
            values: Dictionary mapping metric names to values.
            step: Training step/iteration.
        """
        self._writer.add_scalars(main_tag, values, step)

    def log_training_metrics(
        self,
        actor_loss: float,
        critic_loss: float,
        # mirror_loss: float,
        # imitation_loss: float,
        mean_reward: float,
        mean_ep_len: float,
        mean_noise_std: float,
        step: int,
    ) -> None:
        """Log standard training metrics.

        Args:
            actor_loss: Actor network loss.
            critic_loss: Critic network loss.
            mirror_loss: Mirror symmetry loss.
            imitation_loss: Imitation learning loss.
            mean_reward: Mean episode reward.
            mean_ep_len: Mean episode length.
            mean_noise_std: Mean action noise standard deviation.
            step: Training step/iteration.
        """
        self._writer.add_scalar("Loss/actor", actor_loss, step)
        self._writer.add_scalar("Loss/critic", critic_loss, step)
        # self._writer.add_scalar("Loss/mirror", mirror_loss, step)
        # self._writer.add_scalar("Loss/imitation", imitation_loss, step)
        self._writer.add_scalar("Train/mean_reward", mean_reward, step)
        self._writer.add_scalar("Train/mean_episode_length", mean_ep_len, step)
        self._writer.add_scalar("Train/mean_noise_std", mean_noise_std, step)

    def log_eval_metrics(
        self,
        mean_reward: float,
        mean_ep_len: float,
        step: int,
    ) -> None:
        """Log evaluation metrics.

        Args:
            mean_reward: Mean evaluation episode reward.
            mean_ep_len: Mean evaluation episode length.
            step: Training step/iteration.
        """
        self._writer.add_scalar("Eval/mean_reward", mean_reward, step)
        self._writer.add_scalar("Eval/mean_episode_length", mean_ep_len, step)

    def log_timing_metrics(
        self,
        fps: float,
        sample_time: float,
        optimize_time: float,
        total_time: float,
        step: int,
    ) -> None:
        """Log timing/performance metrics.

        Args:
            fps: Frames (steps) per second.
            sample_time: Time spent sampling trajectories (seconds).
            optimize_time: Time spent in optimizer (seconds).
            total_time: Total elapsed training time (seconds).
            step: Training step/iteration.
        """
        self._writer.add_scalar("Time/fps", fps, step)
        self._writer.add_scalar("Time/sample_time", sample_time, step)
        self._writer.add_scalar("Time/optimize_time", optimize_time, step)
        self._writer.add_scalar("Time/total_elapsed", total_time, step)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Close the logger and release resources."""
        self._writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures logger is closed."""
        self.close()
        return False

class VisLogger(object):
    def __init__(
        self,
        logging_freq_hz: int,
        output_folder: str = "results",
        duration_sec: int = 0,
    ):
        """
        Parameters
        ----------
        logging_freq_hz : int
            记录频率 (Hz)
        output_folder : str, optional
            输出文件夹路径
        duration_sec : int, optional
            预计仿真时长（用于预分配数组，提高性能）
        """
        self.OUTPUT_FOLDER = output_folder
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.PREALLOCATED_ARRAYS = duration_sec > 0
        
        # 计数器
        self.counter = 0
        
        # 时间戳
        max_steps = duration_sec * logging_freq_hz if duration_sec > 0 else 1000
        self.timestamps = np.zeros(max_steps)
        
        # 状态数据: 20维
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, qx, qy, qz, qw, target_x, target_y, target_z, thrust]
        self.states = np.zeros((20, max_steps))
        
        # 控制数据: 12维
        # [thrust, rate_x, rate_y, rate_z, 0, 0, 0, 0, 0, 0, 0, 0]
        self.controls = np.zeros((12, max_steps))
        
        # 结束标记
        self.crashed_step = 0  # 坠毁步数
        self.finished_step = 0  # 完成步数
        self.end_step = 0  # 最终结束步数

    ################################################################################

    def log(self, timestamp: float, state: np.ndarray, control: np.ndarray = None):
        """
        记录单个时间步的数据

        Parameters
        ----------
        timestamp : float
            时间戳 (秒)
        state : np.ndarray
            状态向量，形状 (20,)
            [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, qx, qy, qz, qw, target_x, target_y, target_z, thrust]
        control : np.ndarray, optional
            控制向量，形状 (12,)，默认全零
        """
        if control is None:
            control = np.zeros(12)
        
        if len(state) != 20 or len(control) != 12:
            print("[ERROR] in SingleDroneLogger.log(): invalid state or control length")
            return
        
        # 如果数组不够大，扩展数组
        if self.counter >= self.timestamps.shape[0]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros(1000)))
            self.states = np.concatenate((self.states, np.zeros((20, 1000))), axis=1)
            self.controls = np.concatenate((self.controls, np.zeros((12, 1000))), axis=1)
            current_idx = self.counter
        
        # 如果没有预分配，使用最后一个位置
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[0] > self.counter:
            current_idx = self.timestamps.shape[0] - 1
        else:
            current_idx = self.counter
        
        # 记录数据
        self.timestamps[current_idx] = timestamp
        self.states[:, current_idx] = state
        self.controls[:, current_idx] = control
        self.counter += 1

    ################################################################################

    def set_crashed(self, step: int = None):
        """标记坠毁"""
        if step is None:
            step = self.counter
        self.crashed_step = step
        self._finalize_steps()

    ################################################################################

    def set_finished(self, step: int = None):
        """标记完成"""
        if step is None:
            step = self.counter
        self.finished_step = step
        self._finalize_steps()

    ################################################################################

    def _finalize_steps(self):
        """计算最终结束步数"""
        max_steps = self.counter
        
        # 0表示未发生，使用max_steps作为占位符
        finished = max_steps if self.finished_step == 0 else self.finished_step
        crashed = max_steps if self.crashed_step == 0 else self.crashed_step
        
        # 最终结束步数是两者中较小的
        self.end_step = min(finished, crashed)

    ################################################################################

    def save(self, comment: str = ""):
        """保存为 .npy 文件"""
        filename = os.path.join(
            self.OUTPUT_FOLDER,
            f"save-flight-{comment}-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}.npz"
        )
        np.savez(
            filename,
            timestamps=self.timestamps[:self.counter],
            states=self.states[:, :self.counter],
            controls=self.controls[:, :self.counter],
            crashed_step=self.crashed_step,
            finished_step=self.finished_step,
        )
        print(f"[INFO] Log saved to: {filename}")
        return filename

    ################################################################################

    def save_as_csv(self, comment: str = "", save_timestamps: bool = False) -> str:
        """保存为 CSV 文件"""
        current_time = f"-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}" if save_timestamps else ""
        csv_dir = os.path.join(self.OUTPUT_FOLDER, f"save-flight-{comment}{current_time}")
        os.makedirs(csv_dir, exist_ok=True)
        
        end_idx = self.end_step if self.end_step > 0 else self.counter
        t = np.arange(0, end_idx) / self.LOGGING_FREQ_HZ
        
        data_dict = {
            "time": t,
            "x": self.states[0, :end_idx],
            "y": self.states[1, :end_idx],
            "z": self.states[2, :end_idx],
            "vx": self.states[3, :end_idx],
            "vy": self.states[4, :end_idx],
            "vz": self.states[5, :end_idx],
            "roll": self.states[6, :end_idx],
            "pitch": self.states[7, :end_idx],
            "yaw": self.states[8, :end_idx],
            "wx": self.states[9, :end_idx],
            "wy": self.states[10, :end_idx],
            "wz": self.states[11, :end_idx],
            "qw": self.states[15, :end_idx],
            "qx": self.states[12, :end_idx],
            "qy": self.states[13, :end_idx],
            "qz": self.states[14, :end_idx],
            "target_x": self.states[16, :end_idx],
            "target_y": self.states[17, :end_idx],
            "target_z": self.states[18, :end_idx],
            "thrust": self.states[19, :end_idx],
            "input_T": self.controls[0, :end_idx],
            "input_p": self.controls[1, :end_idx],
            "input_q": self.controls[2, :end_idx],
            "input_r": self.controls[3, :end_idx],
        }
        
        df = pd.DataFrame(data_dict)
        csv_path = os.path.join(csv_dir, f"{comment}_flight_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV saved to: {csv_path}")
        return csv_dir

    ################################################################################

    def plot(self, save_path: str = None):
        """
        绘制所有状态和控制曲线

        Parameters
        ----------
        save_path : str, optional
            如果提供，保存图片到此路径
        """
        fig, axs = plt.subplots(10, 2, figsize=(20, 18))
        fig.suptitle("Drone Simulation Logs", fontsize=16, fontweight="bold")
        
        end_idx = self.end_step if self.end_step > 0 else self.counter
        t = np.arange(0, end_idx) / self.LOGGING_FREQ_HZ

        # 绘图配置: (y标签, 数据索引, 目标索引)
        plot_configs = [
            # 第0列: 位置、姿态
            [
                ("x (m)", 0, 16),
                ("y (m)", 1, 17),
                ("z (m)", 2, 18),
                ("Roll (rad)", 6, None),
                ("Pitch (rad)", 7, None),
                ("Yaw (rad)", 8, None),
                ("q w", 15, None),
                ("q x", 12, None),
                ("q y", 13, None),
                ("q z", 14, None),
            ],
            # 第1列: 速度、角速度、控制
            [
                ("vx (m/s)", 3, None),
                ("vy (m/s)", 4, None),
                ("vz (m/s)", 5, None),
                ("wx (rad/s)", 9, None),
                ("wy (rad/s)", 10, None),
                ("wz (rad/s)", 11, None),
                ("Input Thrust", 0, None),
                ("Input p", 1, None),
                ("Input q", 2, None),
                ("Input r", 3, None),
            ],
        ]

        for col_idx, col_configs in enumerate(plot_configs):
            for row_idx, (y_label, data_idx, target_idx) in enumerate(col_configs):
                ax = axs[row_idx, col_idx]
                
                # 选择数据源
                if col_idx == 0 or row_idx < 6:
                    data = self.states[data_idx, :end_idx]
                else:
                    data = self.controls[data_idx - 6, :end_idx]
                
                # 绘制目标（如果有）
                if target_idx is not None:
                    ax.plot(t, self.states[target_idx, :end_idx], 'k:', label='Target', alpha=0.7)
                
                # 绘制主数据
                ax.plot(t, data, '#1f77b4', label='Drone', linewidth=1.5)
                
                # 标记完成或坠毁
                if row_idx < 3 and col_idx == 0:
                    if self.finished_step > 0 and self.finished_step <= end_idx:
                        finish_t = self.finished_step / self.LOGGING_FREQ_HZ
                        finish_val = self.states[row_idx, self.finished_step - 1]
                        ax.plot(finish_t, finish_val, 'g*', markersize=12, label='Finished')
                    
                    if self.crashed_step > 0 and self.crashed_step <= end_idx:
                        crash_t = self.crashed_step / self.LOGGING_FREQ_HZ
                        crash_val = self.states[row_idx, self.crashed_step - 1]
                        ax.plot(crash_t, crash_val, 'rX', markersize=10, label='Crashed')
                
                # 设置标签
                if row_idx == len(col_configs) - 1:
                    ax.set_xlabel("Time (s)", fontsize=10)
                else:
                    ax.tick_params(labelbottom=False)
                
                ax.set_ylabel(y_label, fontsize=10)
                ax.grid(True, alpha=0.3)
                if row_idx == 0:
                    ax.legend(loc='upper right', fontsize=9)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Plot saved to: {save_path}")
        
        plt.show()