import os
import uuid
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'eeg', 'dat', 'txt', 'bin', 'xls', 'xlsx', 'csv'}  # 脑电波文件的允许扩展名
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # 允许的图片扩展名
UPLOAD_FOLDER = 'static/upload'

def allowed_file(filename, file_type='brain_wave'):
    """检查文件扩展名是否允许上传
    
    Args:
        filename: 文件名
        file_type: 文件类型，可选值：'brain_wave'或'avatar'
    """
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'brain_wave':
        return ext in ALLOWED_EXTENSIONS
    elif file_type == 'avatar':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return False

def generate_unique_filename(filename):
    """生成唯一的文件名，避免文件覆盖"""
    # 获取文件扩展名
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    # 生成基于时间和随机UUID的文件名
    unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    if ext:
        return f"{unique_name}.{ext}"
    return unique_name

def save_uploaded_file(file, subfolder='', file_type='brain_wave'):
    """保存上传的文件到指定目录，并返回保存路径"""
    if not file or not allowed_file(file.filename, file_type):
        return None
    
    # 创建目标目录（如果不存在）
    target_dir = os.path.join(UPLOAD_FOLDER, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    
    # 生成安全的文件名
    filename = secure_filename(file.filename)
    unique_filename = generate_unique_filename(filename)
    
    # 保存文件
    file_path = os.path.join(target_dir, unique_filename)
    file.save(os.path.join(os.getcwd(), file_path))
    
    return file_path

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compare_brain_wave_files(file1_path, file2_path):
    """比较两个脑电波文件是否匹配
    
    实际项目中，这里应该实现专门的脑电波特征比对算法
    现在简单使用文件哈希值比较作为示例
    """
    # 简单实现：比较文件哈希值
    hash1 = calculate_file_hash(file1_path)
    hash2 = calculate_file_hash(file2_path)
    
    # 在实际项目中，可能需要更复杂的比较逻辑和阈值设置
    return hash1 == hash2 