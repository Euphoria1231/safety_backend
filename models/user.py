from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt
import uuid
import os
import random
import string

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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    DEFAULT_PASSWORD = "123456"  # 默认密码

    def __init__(self, username, password=None, email=None, avatar=None, brain_wave_file=None):
        self.username = username
        self.set_password(password or self.DEFAULT_PASSWORD)
        self.email = email
        self.avatar = avatar
        self.brain_wave_file = brain_wave_file

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
        """根据脑电波文件查找用户（随机概率版本）
        
        有50%的概率返回第一个用户，50%的概率返回None
        """
        # 生成0-1之间的随机数
        if random.random() < 0.5:
            return cls.query.first()
        return None

    @classmethod
    def create_with_brain_wave(cls, brain_wave_file_path):
        """通过脑电波文件创建新用户，使用默认密码"""
        # 生成随机用户名
        username = cls.generate_username()
        
        # 创建新用户，使用默认密码
        new_user = cls(
            username=username,
            password=cls.DEFAULT_PASSWORD,  # 使用默认密码
            brain_wave_file=brain_wave_file_path
        )
        
        # 返回新用户和默认密码
        return new_user, cls.DEFAULT_PASSWORD

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