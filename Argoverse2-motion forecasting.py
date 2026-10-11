import pandas as pd
import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
import wandb
from datetime import datetime
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

DATA_ROOT = Path("/kaggle/input/datasets/jiabaoqiang/argoverse2-500-subset/argoverse2_500_subset/")

scenario_dir = sorted(DATA_ROOT.iterdir())[0]
scenario_path = next(scenario_dir.glob("*.parquet"))

scenario = load_argoverse_scenario_parquet(scenario_path)

# ------------------------------------------------------------
from av2.map.map_api import ArgoverseStaticMap

def load_map_for_scenario(scenario_path: Path):
    """
    加载并查看场景对应的地图

    Args:
        scenario_path: 场景parquet文件的路径

    Returns:
        static_map: ArgoverseStaticMap对象
    """
    # 从场景文件名中提取scenario_id
    # 例如: "0000b0f9-99f9-4a1f-a231-5be9e4c523f7_scenario.parquet" -> "0000b0f9-99f9-4a1f-a231-5be9e4c523f7"
    scenario_id = scenario_path.stem.split('_')[-1]
    map_json_path = scenario_path.parent / f"log_map_archive_{scenario_id}.json"

    # 检查地图文件是否存在
    if not map_json_path.exists():
        print(f"警告: 地图文件不存在: {map_json_path}")
        return None

    # 加载静态地图
    static_map = ArgoverseStaticMap.from_json(map_json_path)

    # 查看地图包含的元素
    print(f"\n{'='*60}")
    print(f"场景ID: {scenario_id}")
    print(f"地图文件: {map_json_path}")
    print(f"{'='*60}")

    # 使用正确的属性名称
    lane_segments = static_map.vector_lane_segments
    drivable_areas = static_map.vector_drivable_areas
    pedestrian_crossings = static_map.vector_pedestrian_crossings

    print(f"车道段 (lane_segments) 数量: {len(lane_segments)}")
    print(f"可行驶区域 (drivable_areas) 数量: {len(drivable_areas)}")
    print(f"人行横道 (pedestrian_crossings) 数量: {len(pedestrian_crossings)}")

    # 获取地图的边界，用于可视化或坐标归一化
    all_lane_points = []

    # 使用 get_lane_segment_centerline 方法获取中心线
    for lane_id in lane_segments.keys():
        try:
            centerline = static_map.get_lane_segment_centerline(lane_id)
            # centerline 应该是一个 Nx3 的 numpy 数组，包含 (x, y, z) 坐标
            all_lane_points.extend(centerline)
        except Exception as e:
            print(f"警告: 无法获取车道 {lane_id} 的中心线: {e}")

    # 转换为numpy数组后计算min/max
    if all_lane_points:
        all_lane_points = np.array(all_lane_points)
        x_min, y_min, z_min = all_lane_points.min(axis=0)
        x_max, y_max, z_max = all_lane_points.max(axis=0)

        print(f"\n地图边界:")
        print(f"  X范围: [{x_min:.2f}, {x_max:.2f}] m")
        print(f"  Y范围: [{y_min:.2f}, {y_max:.2f}] m")
        print(f"  Z范围: [{z_min:.2f}, {z_max:.2f}] m")
        print(f"  地图尺寸: {x_max - x_min:.2f} x {y_max - y_min:.2f} m")

        # 统计车道类型
        lane_types = {}
        intersection_count = 0
        for lane_id in lane_segments.keys():
            lane_seg = lane_segments[lane_id]
            lane_type = lane_seg.lane_type
            lane_types[lane_type] = lane_types.get(lane_type, 0) + 1
            if lane_seg.is_intersection:
                intersection_count += 1

        print(f"\n车道类型统计:")
        for lane_type, count in lane_types.items():
            print(f"  {lane_type}: {count}")
        print(f"交叉口内的车道段: {intersection_count}")

        # 显示部分车道段ID作为示例
        lane_ids = list(lane_segments.keys())
        print(f"\n示例车道段ID (前5个):")
        for i, lane_id in enumerate(lane_ids[:5]):
            lane_seg = lane_segments[lane_id]
            print(f"  [{i+1}] ID: {lane_id}, 类型: {lane_seg.lane_type}, 交叉口: {lane_seg.is_intersection}")

        # 显示车道拓扑关系示例
        print(f"\n车道拓扑关系示例 (前3个车道段):")
        for i, lane_id in enumerate(lane_ids[:3]):
            lane_seg = lane_segments[lane_id]
            successors = lane_seg.successors if lane_seg.successors else []
            predecessors = lane_seg.predecessors if lane_seg.predecessors else []
            left_neighbor = lane_seg.left_neighbor_id
            right_neighbor = lane_seg.right_neighbor_id
            print(f"  [{i+1}] 车道 {lane_id}:")
            print(f"      前驱车道: {predecessors if predecessors else '无'}")
            print(f"      后继车道: {successors if successors else '无'}")
            print(f"      左侧邻居: {left_neighbor if left_neighbor else '无'}")
            print(f"      右侧邻居: {right_neighbor if right_neighbor else '无'}")
    else:
        print("\n警告: 无法提取车道点数据")

    print(f"{'='*60}\n")

    return static_map

# 加载示例场景的地图
static_map_sample = load_map_for_scenario(scenario_path)

# ------------------------------------------------------------------------------------

# TODO：真正训练的时候，改为使用多进程和懒加载数据，每次读取10000个数据进内存，多个CPU worker交替工作，提前预加载数据到CPU
import torch
from torch.utils.data import Dataset

class AV2Dataset(Dataset):

    def __init__(self, root, max_files=None):
        """
        Args:
            root: 数据集根目录
            max_files: 最大文件数，None表示使用全部文件
        """
        root_path = Path(root)
        self.files = list(root_path.glob("*/*.parquet")) #这一句耗时比较长
        
        if max_files is not None and max_files < len(self.files):
            self.files = self.files[:max_files]
        
        print(f"找到 {len(self.files)} 个数据文件")
        
        # 预加载数据到内存，避免训练时重复I/O
        print(f"开始预加载数据到内存...")
        self.data_cache = []
        for i, file in enumerate(self.files):
            if i % 50 == 0:
                print(f"加载进度: {i}/{len(self.files)}")
            try:
                scenario = load_argoverse_scenario_parquet(file)
                focal = next(t for t in scenario.tracks if t.track_id == scenario.focal_track_id)
                traj = np.array([s.position for s in focal.object_states])
                vel = np.array([s.velocity for s in focal.object_states])
                # 确保轨迹长度足够
                if len(traj) >= 110:  # 50历史 + 60未来
                    # 平移 + 旋转
                    # 将绝对坐标值转换成相对坐标值，第5秒的位置设置为(0,0)
                    origin = traj[49]  # 第5秒的位置作为原点
                    history_pos = traj[:50] - origin
                    future_pos = traj[50:110] - origin
                    history_vel = vel[:50]
                    future_vel = vel[50:110]

                    # 计算最后一段运动方向（只用最后10帧）
                    direction = history_pos[-1] - history_pos[-11]
                    displacement = np.linalg.norm(direction)

                    # 如果位移接近零，不进行旋转，直接使用平移后的数据
                    if displacement < 1e-2:
                        self.data_cache.append((history_pos, history_vel, future_pos))
                    else:
                        # 计算方向角
                        theta = np.arctan2(direction[1], direction[0])

                        # 构建旋转矩阵（旋转 -theta）
                        R = np.array([
                            [np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]
                        ])

                        # 对历史和未来轨迹都进行旋转
                        history_pos_rot = history_pos @ R.T
                        future_pos_rot = future_pos @ R.T
                        # 速度向量也需要旋转
                        history_vel_rot = history_vel @ R.T
                        future_vel_rot = future_vel @ R.T

                        self.data_cache.append((history_pos_rot, history_vel_rot, future_pos_rot))
            except Exception as e:
                print(f"跳过文件 {file}: {e}")
        
        print(f"✅ 成功加载 {len(self.data_cache)} 个有效样本到内存")

        # 计算标准化参数：分别计算位置和速度的scale
        all_pos = []
        all_vel = []
        for history_pos, history_vel, future_pos in self.data_cache:
            all_pos.append(history_pos)
            all_pos.append(future_pos)
            all_vel.append(history_vel)
        all_pos = np.concatenate(all_pos, axis=0)
        all_vel = np.concatenate(all_vel, axis=0)
        self.pos_scale = np.std(all_pos, axis=0).max()  # 取x,y标准差的最大值作为位置scale
        self.vel_scale = np.std(all_vel, axis=0).max()  # 取vx,vy标准差的最大值作为速度scale
        print(f"标准化参数 pos_scale: {self.pos_scale:.4f}, vel_scale: {self.vel_scale:.4f}")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        # 直接从内存缓存读取，避免重复I/O
        history_pos, history_vel, future_pos = self.data_cache[idx]
        # 分别标准化位置和速度，然后拼接
        history_normalized = np.concatenate([history_pos / self.pos_scale, history_vel / self.vel_scale], axis=1)
        future_normalized = future_pos / self.pos_scale
        return (
            torch.FloatTensor(history_normalized),
            torch.FloatTensor(future_normalized)
        )


# ------------------------------------------------------------------------------------

import torch.nn as nn

class LSTMPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        # 多层LSTM编码器，增加dropout
        self.encoder = nn.LSTM(
            input_size=4,  # x, y, vx, vy
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )

        # 添加全连接层增强表达能力
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.decoder = nn.Linear(256, 60*2)

    def forward(self, x):

        _, (h, _) = self.encoder(x)

        # 使用最后一层的隐藏状态
        h = h[-1]

        # 通过全连接层
        out = self.fc1(h)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout2(out)

        out = self.decoder(out)

        return out.reshape(-1, 60, 2)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"可用GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

model = LSTMPredictor().to(device)

# 如果有多个GPU，使用DataParallel进行数据并行
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
    model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

def ADE(pred, gt):
    return torch.norm(pred-gt, dim=-1).mean()

def ADE_at_t(pred, gt, t):
    """计算到第t帧的ADE (0-indexed, t=9对应1s)"""
    return torch.norm(pred[:, :t+1] - gt[:, :t+1], dim=-1).mean()

def FDE(pred, gt):
    return torch.norm(pred[:,-1]-gt[:,-1], dim=-1).mean()

# 创建数据加载器
from torch.utils.data import random_split, DataLoader

# 记录训练指标
train_losses = []
val_losses = []
val_ades = []
val_fdes = []
epochs = []
try:
    # 检查数据集是否能加载
    dataset = AV2Dataset(DATA_ROOT)
    print(f"数据集大小: {len(dataset)}")
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # random_split 创建的是索引的视图，不是数据副本
    # train_dataset 和 val_dataset 只是引用 dataset 的不同索引范围，原始数据只有一份，所以不会有两份数据
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    # 根据GPU数量调整batch_size，保持每个GPU的batch_size为64
    single_batch_size = 64
    all_batch_size = single_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else single_batch_size  
    train_loader = DataLoader(train_dataset, batch_size=all_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=all_batch_size, shuffle=False, num_workers=2)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 初始化wandb
    wandb.init(
        project="argoverse2-trajectory-prediction",
        name=f"lstm-predictor-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        dir="/kaggle/working/wandb",
        config={
            # 数据相关
            "data_root": str(DATA_ROOT),
            "max_files": None,
            # 模型相关
            "learning_rate": 1e-3,
            "epochs": 150,
            "batch_size": single_batch_size,
            "architecture": "LSTM",
            "input_size": 4,  # x, y, vx, vy
            "hidden_size": 256,
            "num_layers": 3,
            "dropout": 0.3,
            # 训练相关
            "dataset": "Argoverse2",
            "optimizer": "Adam",
            "scheduler": "StepLR",
            # 其他
            "comment": "增加速度进行训练"
        }
    )
    
    # 训练循环
    num_epochs = 150
    best_val_ade = float('inf')
    patience = 100000  #暂时关闭此功能  #Early Stopping：验证loss不下降10个epoch就停止
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_ade = 0
        val_fde = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                # 反标准化后再计算ADE/FDE
                pred_denorm = pred * dataset.pos_scale
                batch_y_denorm = batch_y * dataset.pos_scale
                val_ade += ADE(pred_denorm, batch_y_denorm).item()
                val_fde += FDE(pred_denorm, batch_y_denorm).item()
        
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ade = val_ade / len(val_loader)
        avg_val_fde = val_fde / len(val_loader)

        # 记录训练指标
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ades.append(avg_val_ade)
        val_fdes.append(avg_val_fde)
        epochs.append(epoch)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_ade": avg_val_ade,
            "val_fde": avg_val_fde,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch}: train Loss {avg_train_loss:.4f}, val Loss {avg_val_loss:.4f}, val ADE {avg_val_ade:.4f}, val FDE {avg_val_fde:.4f}")
        
        # 保存最佳模型到可写目录
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            no_improve_count = 0
            # 处理DataParallel包装的模型
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '/kaggle/working/best_model.pth')
            wandb.save('/kaggle/working/best_model.pth')
            print(f"保存最佳模型，验证ADE损失: {best_val_ade:.4f}")
        else:
            no_improve_count += 1
            print(f"验证损失未改善，计数: {no_improve_count}/{patience}")
            
            # Early stopping检查
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"最佳验证ADE损失: {best_val_ade:.4f}")
                break
    
    print(f"训练完成！最佳验证ADE损失: {best_val_ade:.4f}")
    wandb.finish()
    
except Exception as e:
    print(f"训练过程中出现错误: {e}")
    print("请检查数据路径是否正确，或数据集是否已正确下载")
    # 确保wandb被正确关闭
    if 'wandb' in locals():
        wandb.finish()

    
# -------------------------------------------------------------------------------

# 绘制训练曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

# 绘制Loss曲线
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, val_losses, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# 绘制ADE曲线
plt.subplot(2, 2, 2)
plt.plot(epochs, val_ades, 'g-', label='Val ADE')
plt.xlabel('Epoch')
plt.ylabel('ADE')
plt.title('Validation ADE')
plt.legend()
plt.grid(True)

# 绘制FDE曲线
plt.subplot(2, 2, 3)
plt.plot(epochs, val_fdes, 'm-', label='Val FDE')
plt.xlabel('Epoch')
plt.ylabel('FDE')
plt.title('Validation FDE')
plt.legend()
plt.grid(True)

# 绘制ADE和FDE对比
plt.subplot(2, 2, 4)
plt.plot(epochs, val_ades, 'g-', label='ADE')
plt.plot(epochs, val_fdes, 'm-', label='FDE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('ADE vs FDE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 使用最佳模型计算不同时长的ADE
print("\n使用最佳模型计算不同预测时长的ADE...")

# 加载最佳模型
best_model = LSTMPredictor().to(device)
best_model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
best_model.eval()

# 计算不同时长的ADE
ade_at_t = {f"ADE@{t}s": 0 for t in range(1, 7)}
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = best_model(batch_x)
        # 反标准化
        pred_denorm = pred * dataset.pos_scale
        batch_y_denorm = batch_y * dataset.pos_scale
        # 计算不同时长的ADE (每秒10帧)
        for t in range(1, 7):
            ade_at_t[f"ADE@{t}s"] += ADE_at_t(pred_denorm, batch_y_denorm, t*10-1).item()

# 计算平均值
avg_ade_at_t = {f"ADE@{t}s": ade_at_t[f"ADE@{t}s"] / len(val_loader) for t in range(1, 7)}

print("\n最佳模型在不同预测时长的ADE:")
for t in range(1, 7):
    print(f"  ADE@{t}s: {avg_ade_at_t[f'ADE@{t}s']:.4f} m")

# 绘制不同时长的ADE
plt.figure(figsize=(6, 5))

# ADE@t 增长趋势
plt.subplot(1, 1, 1)
plt.plot(time_points, ade_values, 'o-', color='#FF6B6B', linewidth=0.5, markersize=4)
plt.xlabel('Prediction Horizon')
plt.ylabel('ADE (m)')
plt.title('ADE Growth with Prediction Horizon')
plt.grid(True)

plt.tight_layout()
plt.show()



# -----------------------------------------------------
# 绘制真实轨迹 vs 预测轨迹 绘制真实轨迹 vs 预测轨迹，随机挑10 个 validation samples，画：
# 历史轨迹：蓝色
# 真实未来：绿色
# 预测未来：红色

import random

# 加载最佳模型
model_to_load = LSTMPredictor().to(device)
model_to_load.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
model_to_load.eval()

# 随机选择10个验证样本
num_samples = 10
val_indices = random.sample(range(len(val_dataset)), num_samples)

fig, axes = plt.subplots(5, 2, figsize=(8, 20))
axes = axes.flatten()

with torch.no_grad():
    for i, idx in enumerate(val_indices):
        # 获取数据
        history_normalized, future_normalized = val_dataset[idx]
        history_pos = history_normalized[:50, :2].numpy() * dataset.pos_scale

        # 获取真实未来轨迹
        true_future = future_normalized.numpy() * dataset.pos_scale

        # 预测未来轨迹
        history_input = history_normalized.unsqueeze(0).to(device)
        pred_future = model_to_load(history_input).squeeze(0).cpu().numpy() * dataset.pos_scale

        # 绘制
        ax = axes[i]
        ax.plot(history_pos[:, 0], history_pos[:, 1], 'b.', label='History', linewidth=0.5)
        ax.plot(true_future[:, 0], true_future[:, 1], 'g.', label='True Future', linewidth=0.5)
        ax.plot(pred_future[:, 0], pred_future[:, 1], 'r-', label='Predicted Future', linewidth=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

plt.tight_layout()
plt.show()






