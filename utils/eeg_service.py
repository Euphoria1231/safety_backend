import sys
import os
import torch
import numpy as np
import sqlite3
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import random

# 添加algorithm文件夹到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_path = os.path.join(current_dir, '..', 'algorithm', 'EEG-main')
sys.path.append(algorithm_path)

from Model_3 import DCNN


def load_model_DCNN(model_path, num_classes):
    """
    加载模型,并使用GPU设备
    返回模型和GPU设备
    """
    model = DCNN(num_classes=num_classes)

    # 安全加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def preprocess_data(eeg_data):
    """
    对EEG数据进行预处理,预处理为数据加维度,这样才能作为特征(或说密码)提取的对象
    """
    # 确保输入是(16, 80)的numpy数组
    assert eeg_data.shape == (16, 80), f"需要(16,80)的输入，但得到{eeg_data.shape}"

    return torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0)


class SQLiteDatabaseManager:
    """
    SQLite数据库管理器，用于EEG特征存储
    """
    def __init__(self, db_path="eeg_features.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()

    def _init_database(self):
        """初始化数据库和表结构"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
            
            # 创建EEG特征表
            cursor = self.connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS eeg_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    features BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.connection.commit()
            print("SQLite数据库初始化成功")
        except Exception as e:
            print(f"SQLite数据库初始化失败: {e}")
            raise

    def connect(self):
        """连接数据库（兼容原接口）"""
        if not self.connection:
            self._init_database()

    def insert_data(self, table, data):
        """
        插入数据到指定表
        """
        try:
            cursor = self.connection.cursor()
            
            if table == "user_info_2":
                # 兼容原始接口，将数据插入到eeg_features表
                cursor.execute(
                    "INSERT OR REPLACE INTO eeg_features (user_id, features) VALUES (?, ?)",
                    (data['user'], data['pwd'])
                )
                self.connection.commit()
                print(f"EEG特征数据插入成功，用户: {data['user']}")
                return cursor.lastrowid
            else:
                raise ValueError(f"不支持的表名: {table}")
                
        except Exception as e:
            print(f"插入数据错误: {e}")
            raise

    def execute_query(self, query_sql, params=None):
        """
        执行查询操作
        """
        try:
            cursor = self.connection.cursor()
            
            if "user_info_2" in query_sql:
                # 兼容原始查询，从eeg_features表查询
                cursor.execute("SELECT user_id as user, features as pwd FROM eeg_features")
                rows = cursor.fetchall()
                # 转换为字典格式
                return [dict(row) for row in rows]
            else:
                if params:
                    cursor.execute(query_sql, params)
                else:
                    cursor.execute(query_sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"查询数据错误: {e}")
            raise


class EEGProcessor_DCNN:
    """
    主要是模型与数据之间的操作
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model_DCNN(model_path, num_classes=109)
        self.templates = {}  # 格式: {user_id: [feature_vec1, feature_vec2,...]}

    def extract_features(self, npz_path):
        """
        从npz文件提取特征,默认提取五个特征
        """
        data = np.load(npz_path)
        features = []
        available_keys = [k for k in data.files if data[k].shape == (16, 80)]

        # 如果可用的键少于5个，就用所有可用的键
        num_samples = min(5, len(available_keys))
        selected_keys = random.sample(available_keys, num_samples)

        for key in selected_keys:
            try:
                eeg = data[key]
                tensor = torch.from_numpy(eeg).float().to(self.device).unsqueeze(0)

                with torch.no_grad():
                    feature = self.model.extract_features(tensor).cpu().numpy()[0]
                    features.append(feature)
            except Exception as e:
                print(f"处理片段 {key} 时出错: {str(e)}")

        if not features:
            raise ValueError(f"未找到有效EEG片段: {npz_path}")
        return np.mean(features, axis=0)


class EEGAuthSystem_DCNN:
    """
    将前面两个类:数据库和模型数据处理放一起
    """
    def __init__(self, model_path):
        self.processor = EEGProcessor_DCNN(model_path)
        self.db = SQLiteDatabaseManager()

    def register_user(self, user_id, npz_path):
        """完整的用户注册流程"""
        try:
            # 提取特征数据
            feature = self.processor.extract_features(npz_path)

            # 将压缩后的特征数据存储到数据库
            record_id = self.db.insert_data(
                table="user_info_2",
                data={
                    "user": user_id,
                    "pwd": feature.tobytes()
                }
            )

            return {
                "status": "success",
                "user_id": user_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def verify(self, npz_path, threshold=0.85):
        """
        用于验证传入EEG数据是否在数据库中,身份验证;置信度为0.85,可更改
        若成功,返回用户名称(id);失败返回"unknown"
        """
        try:
            features = self.processor.extract_features(npz_path).squeeze()

            # 查询用户
            user_info = self.db.execute_query(
                "SELECT user, pwd FROM user_info_2"
            )
            if not user_info:
                return "数据库中没有数据!"

            best_match, best_score = None, -1
            for info in user_info:
                try:
                    feature = np.frombuffer(info['pwd'], dtype=np.float32)

                    # 计算相似度得分
                    score = cosine_similarity([features], [feature])[0][0]
                    if score > best_score:
                        best_score, best_match = score, info['user']

                except Exception as process_error:
                    print(f"处理用户 {info['user']} 时出错: {str(process_error)}")
                    continue

            if best_score >= threshold:
                return best_match
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"验证过程中出现错误: {str(e)}")
            return "Unknown"


class EEGService:
    """
    EEG脑电波认证服务类
    用于处理用户的脑电波文件注册和验证
    """
    
    def __init__(self, model_path: str = "algorithm/EEG-main/models/DCNN_16x80.pth"):
        """
        初始化EEG服务
        
        Args:
            model_path: DCNN模型文件路径
        """
        self.model_path = model_path
        self.auth_system = None
        self._init_auth_system()
    
    def _init_auth_system(self):
        """初始化EEG认证系统"""
        try:
            # 创建EEG认证系统实例
            self.auth_system = EEGAuthSystem_DCNN(self.model_path)
            
            # 连接数据库
            self.auth_system.db.connect()
            print("EEG认证系统初始化成功")
        except Exception as e:
            print(f"EEG认证系统初始化失败: {str(e)}")
            self.auth_system = None
    
    def register_user_with_eeg(self, user_id: str, npz_file_path: str) -> Dict[str, Any]:
        """
        使用EEG数据注册用户
        
        Args:
            user_id: 用户ID
            npz_file_path: NPZ文件路径
            
        Returns:
            注册结果字典，包含状态和消息
        """
        if not self.auth_system:
            return {
                "status": "error",
                "message": "EEG认证系统未初始化"
            }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(npz_file_path):
                return {
                    "status": "error",
                    "message": f"文件不存在: {npz_file_path}"
                }
            
            # 调用EEG认证系统注册用户
            result = self.auth_system.register_user(user_id, npz_file_path)
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"注册失败: {str(e)}"
            }
    
    def verify_user_with_eeg(self, npz_file_path: str, threshold: float = 0.85) -> str:
        """
        使用EEG数据验证用户身份
        
        Args:
            npz_file_path: NPZ文件路径
            threshold: 相似度阈值，默认0.85
            
        Returns:
            用户ID或"Unknown"表示未找到匹配用户
        """
        if not self.auth_system:
            return "Unknown"
        
        try:
            # 检查文件是否存在
            if not os.path.exists(npz_file_path):
                return "Unknown"
            
            # 调用EEG认证系统验证用户
            result = self.auth_system.verify(npz_file_path, threshold)
            return result
            
        except Exception as e:
            print(f"EEG验证失败: {str(e)}")
            return "Unknown"
    
    def convert_file_to_npz(self, file_path: str) -> Optional[str]:
        """
        将其他格式的脑电波文件转换为NPZ格式（如需要）
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            转换后的NPZ文件路径，失败返回None
        """
        # 这里可以实现文件格式转换逻辑
        # 当前假设文件已经是NPZ格式或可以直接使用
        if file_path.endswith('.npz'):
            return file_path
        
        # TODO: 实现其他格式到NPZ的转换
        # 目前直接返回原文件路径，实际项目中需要根据需求实现转换逻辑
        return None
    
    def is_valid_eeg_file(self, file_path: str) -> bool:
        """
        检查是否为有效的EEG文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效的EEG文件
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # 检查文件扩展名
            valid_extensions = ['.npz', '.eeg', '.dat', '.txt', '.bin']
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # 如果是NPZ文件，尝试加载验证
            if file_path.endswith('.npz'):
                try:
                    data = np.load(file_path)
                    # 检查是否包含有效的EEG数据片段
                    valid_keys = [k for k in data.files if data[k].shape == (16, 80)]
                    return len(valid_keys) > 0
                except:
                    return False
            
            return True
            
        except Exception as e:
            print(f"验证EEG文件失败: {str(e)}")
            return False


# 创建全局EEG服务实例
eeg_service = EEGService() 