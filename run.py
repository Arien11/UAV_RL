import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 切换到项目根目录
os.chdir(project_root)

# 导入并运行训练
from train._train import *

if __name__ == '__main__':
    # train
    _ppo = PPO(make_env_fc)
    train_proc = Training(make_env_fc, _ppo)
    model_path = None
    if model_path:
        train_proc.train(200, model_path=model_path)
    else:
        train_proc.train(50)
    
    # eval
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_path = "E:\\UAV_RL\\train\\checkpoints\\checkpoint_iter10.pt"
    # policy = load_policy_for_inference(checkpoint_path, device=device)
    #
    # # 重要：恢复动作缩放参数（这些未保存在 state_dict 中，需手动设置）
    # # 请确保这些值与训练时完全一致
    # policy.action_scale = torch.tensor([2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0], device=device)
    # policy.action_bias = torch.zeros(9, device=device)
    #
    # # 运行测试（渲染模式）
    # test_policy(policy, make_env_fc, num_episodes=3, deterministic=True, render=True)