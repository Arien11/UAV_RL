import os
import numpy as np
from numpy.core.records import fromarrays
from typing import List, Optional, Union
from utils.logger import VisLogger
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from race_utils.RaceVisualizer.RacePlotter import RacePlotter, BasePlotterList
from race_utils.RaceGenerator.GenerationTools import create_state, create_gate

def _data_to_racetrack(data: dict, shape_kwargs: dict, noise_matrix: np.ndarray) -> List[RaceTrack]:
    
    comment = track_data["comment"]
    track_num_drones = 1
    same_track = track_data.get("same_track", True) # 是否使用相同轨迹
    repeat_lap = track_data.get("repeat_lap", 1)   # 重复多圈数

    # 读取航点
    start_points = np.array(track_data["start_points"], dtype=float).reshape((1, -1, 3))
    end_points = np.array(track_data["end_points"], dtype=float).reshape((1, -1, 3))
    waypoints = np.array(track_data["waypoints"], dtype=float).reshape((1, -1, 3))
    
    # 生成完整航点序列
    if repeat_lap == 1:
        main_segments = waypoints
    else:
        main_segments = np.tile(waypoints, (1, repeat_lap, 1))  # 重复多圈
    # 拼接起始点和结束点
    if start_points.shape[1] > 1:
        main_segments = np.concatenate((start_points[:, 1:], main_segments), axis=1)
    if end_points.shape[1] > 1:
        main_segments = np.concatenate((main_segments, end_points[:, :-1]), axis=1)
    # 添加噪声
    if noise_matrix is not None:
        main_segments += noise_matrix
    
    racetrack_list = []
    main_seg_len = main_segments.shape[1]
    
    init_state = create_state({"pos": start_points[0]})
    end_state = create_state({"pos": end_points[0]})

    # 创建 RaceTrack 对象
    race_track = RaceTrack(init_state=init_state, end_state=end_state, race_name=f"{comment}_drone1")
    
    # 为每个航点创建门并添加到赛道
    for j in range(main_seg_len):
        gate = create_gate(
            gate_type="SingleBall",
            position=main_segments[0, j],
            stationary=True,
            shape_kwargs=shape_kwargs,
            name=f"{comment}_Gate_{j+1}",
        )
        race_track.add_gate(gate, gate.name)
    
    racetrack_list.append(race_track)

def _logger_to_traj_data(logger: VisLogger) -> List[np.ndarray]:
    """
    Convert the logger data to a list.

    Parameters
    ----------
    logger : Logger
        The logger to convert.

    Returns
    -------
    List[np.ndarray]
        The data list for all drones.

    """
    states = logger.states.copy()

    # define the data type
    dtype = [
        ("t", "f8"),  # time
        ("p_x", "f8"),
        ("p_y", "f8"),
        ("p_z", "f8"),  # position
        ("q_x", "f8"),
        ("q_y", "f8"),
        ("q_z", "f8"),
        ("q_w", "f8"),  # quaternion
        ("v_x", "f8"),
        ("v_y", "f8"),
        ("v_z", "f8"),  # velocity
    ]

    t = np.arange(0, logger.counter) / logger.LOGGING_FREQ_HZ
    data = np.concatenate([
        t.reshape(1, -1),
        states[0:3, :],    # 位置
        states[12:16, :],  # 四元数
        states[3:6, :],     # 速度
    ], axis=0).T
    
    arrays = [data[:, j] for j in range(data.shape[1])]
    data = fromarrays(arrays, dtype=dtype)
    
    if logger.end_step > 0:
        data = data[:logger.end_step]
    # 计算结束时间和坠毁标记
    end_time = logger.end_step / logger.LOGGING_FREQ_HZ if logger.end_step > 0 else None
    crash_effect = logger.crashed_step > 0
    
    return [data], end_time, crash_effect


def create_raceplotter(
    logger,
    track_data: dict,
    shape_kwargs: dict,
    noise_matrix: Optional[np.ndarray] = None,
) -> BasePlotterList:
    """
    为单机版 Logger 创建 RacePlotter
    
    Parameters
    ----------
    logger : SingleDroneLogger
        单机版 Logger 对象
    track_data : dict
        赛道数据
    shape_kwargs : dict
        形状参数
    noise_matrix : np.ndarray, optional
        噪声矩阵
    
    Returns
    -------
    BasePlotterList
        绘图器列表
    """
    # 1.转换 Logger 数据
    data_list, end_time, crash_effect = _logger_to_traj_data(logger)

    # 2.创建赛道
    racetrack_list = _data_to_racetrack(track, noise_matrix)
    
    # 3.创建 RacePlotter绘图器
    raceplotter = RacePlotter(
        traj_file=data_list[0],       # 轨迹数据
        track_file=racetrack_list[0], # 赛道数据
        end_time=end_time,
        crash_effect=crash_effect,
        crash_kwargs={"color": "#1f77b4"},
    )
    
    return BasePlotterList(plotters=[raceplotter])


def load_plotter_track(
    current_dir: Union[str, os.PathLike],
    track_file: Union[str, os.PathLike, RaceTrack],
    plotter: Optional[RacePlotter] = None,
    index: Optional[list] = None,
    plot_track_once: bool = False,
) -> Union[RacePlotter, BasePlotterList]:
    """
    Load the track file.

    Parameters
    ----------
    track_file : str
        The track file to load.
    index : list, optional
        The index of the track file to load, by default None
        If None, load all the track files.

    Returns
    -------
    Union[RacePlotter, BasePlotterList]
        The loaded plotters.

    """
    track_file = os.path.join(current_dir, "gym_drones/assets/Tracks/RaceUtils", f"{track_file}.yaml")
    if isinstance(plotter, RacePlotter):
        plotter.load_track(track_file=track_file, index=index)
    else:
        plotter.load_track(track_file=track_file, index=index, plot_track_once=plot_track_once)
    return plotter
