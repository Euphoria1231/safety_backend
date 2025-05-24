from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt
import uuid
import os
import random
import string
from utils.eeg_service import eeg_service

db = SQLAlchemy()

class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(128), nullable=False)
    avatar = db.Column(db.String(200), nullable=True)
    brain_wave_file = db.Column(db.String(200), nullable=True)
    eeg_user_id = db.Column(db.String(100), nullable=True)  # EEG系统中的用户ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    DEFAULT_PASSWORD = "123456"  # 默认密码

    def __init__(self, username, password=None, email=None, avatar=None, brain_wave_file=None):
        self.username = username
        self.set_password(password or self.DEFAULT_PASSWORD)
        self.email = email
        self.avatar = avatar
        self.brain_wave_file = brain_wave_file
        self.eeg_user_id = None

    def set_password(self, password):
        """设置密码，使用bcrypt加密"""
        if isinstance(password, str):
            password = password.encode('utf-8')
        self.password_hash = bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        """验证密码是否正确"""
        if isinstance(password, str):
            password = password.encode('utf-8')
        return bcrypt.checkpw(password, self.password_hash.encode('utf-8'))

    @staticmethod
    def generate_username():
        """生成随机用户名，格式为User + 随机字符串"""
        random_str = str(uuid.uuid4())[:8]
        return f"User{random_str}"

    @classmethod
    def find_by_brain_wave_file(cls, file_path):
        """
        根据脑电波文件查找用户（使用EEG算法）
        
        Args:
            file_path: 脑电波文件路径
            
        Returns:
            匹配的用户对象或None
        """
        try:
            # 检查文件是否为有效的EEG文件
            if not eeg_service.is_valid_eeg_file(file_path):
                return None
            
            # 尝试转换为NPZ格式（如果需要）
            npz_file_path = eeg_service.convert_file_to_npz(file_path)
            if not npz_file_path:
                # 如果文件不是NPZ格式且无法转换，使用原路径
                npz_file_path = file_path
            
            # 使用EEG算法验证用户身份
            verified_eeg_user_id = eeg_service.verify_user_with_eeg(npz_file_path)
            
            if verified_eeg_user_id != "Unknown" and verified_eeg_user_id != "数据库中没有数据!":
                # 根据EEG用户ID查找对应的用户
                user = cls.query.filter_by(eeg_user_id=verified_eeg_user_id).first()
                return user
            
            return None
            
        except Exception as e:
            print(f"脑电波文件查找用户失败: {str(e)}")
            return None

    @classmethod
    def create_with_brain_wave(cls, brain_wave_file_path):
        """
        通过脑电波文件创建新用户，使用EEG算法注册
        
        Args:
            brain_wave_file_path: 脑电波文件路径
            
        Returns:
            (新用户对象, 默认密码) 或 None
        """
        try:
            # 生成随机用户名
            username = cls.generate_username()
            
            # 生成EEG系统中的用户ID（使用用户名作为EEG用户ID）
            eeg_user_id = f"eeg_{username.lower()}"
            
            # 检查文件是否为有效的EEG文件
            if not eeg_service.is_valid_eeg_file(brain_wave_file_path):
                print(f"无效的EEG文件: {brain_wave_file_path}")
                return None
            
            # 尝试转换为NPZ格式（如果需要）
            npz_file_path = eeg_service.convert_file_to_npz(brain_wave_file_path)
            if not npz_file_path:
                npz_file_path = brain_wave_file_path
            
            # 在EEG系统中注册用户
            eeg_result = eeg_service.register_user_with_eeg(eeg_user_id, npz_file_path)
            
            if eeg_result.get("status") != "success":
                print(f"EEG注册失败: {eeg_result.get('message', '未知错误')}")
                return None
            
            # 创建新用户，使用默认密码
            new_user = cls(
                username=username,
                password=cls.DEFAULT_PASSWORD,
                brain_wave_file=brain_wave_file_path
            )
            new_user.eeg_user_id = eeg_user_id
            
            # 返回新用户和默认密码
            return new_user, cls.DEFAULT_PASSWORD
            
        except Exception as e:
            print(f"创建脑电波用户失败: {str(e)}")
            return None

    def update_brain_wave_file(self, new_brain_wave_file_path):
        """
        更新用户的脑电波文件
        
        Args:
            new_brain_wave_file_path: 新的脑电波文件路径
            
        Returns:
            更新是否成功
        """
        try:
            # 检查新文件是否为有效的EEG文件
            if not eeg_service.is_valid_eeg_file(new_brain_wave_file_path):
                return False
            
            # 如果用户已有EEG用户ID，需要在EEG系统中更新
            if self.eeg_user_id:
                # 尝试转换为NPZ格式（如果需要）
                npz_file_path = eeg_service.convert_file_to_npz(new_brain_wave_file_path)
                if not npz_file_path:
                    npz_file_path = new_brain_wave_file_path
                
                # 重新注册（这会覆盖原有的特征）
                eeg_result = eeg_service.register_user_with_eeg(self.eeg_user_id, npz_file_path)
                
                if eeg_result.get("status") != "success":
                    print(f"EEG更新失败: {eeg_result.get('message', '未知错误')}")
                    return False
            else:
                # 如果用户没有EEG用户ID，创建一个
                self.eeg_user_id = f"eeg_{self.username.lower()}"
                
                npz_file_path = eeg_service.convert_file_to_npz(new_brain_wave_file_path)
                if not npz_file_path:
                    npz_file_path = new_brain_wave_file_path
                
                eeg_result = eeg_service.register_user_with_eeg(self.eeg_user_id, npz_file_path)
                
                if eeg_result.get("status") != "success":
                    print(f"EEG初次注册失败: {eeg_result.get('message', '未知错误')}")
                    return False
            
            # 更新用户的脑电波文件路径
            self.brain_wave_file = new_brain_wave_file_path
            return True
            
        except Exception as e:
            print(f"更新脑电波文件失败: {str(e)}")
            return False

    def to_dict(self):
        """将用户对象转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'avatar': self.avatar,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'has_brain_wave': bool(self.brain_wave_file)
        } 