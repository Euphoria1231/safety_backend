from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models import User, Note, db
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime
from sqlalchemy import func
import bcrypt

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# 管理员账号配置
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123456"  # 实际项目中应该使用环境变量

class AdminLoginSchema(Schema):
    """管理员登录验证模式"""
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class UpdateUserSchema(Schema):
    """更新用户信息验证模式"""
    username = fields.Str(required=False, validate=validate.Length(min=2, max=30))
    email = fields.Email(required=False, allow_none=True)
    is_active = fields.Bool(required=False)

def verify_admin_token():
    """验证管理员令牌"""
    try:
        current_user_id = get_jwt_identity()
        # 检查是否为管理员令牌（以admin_开头）
        if not current_user_id or not current_user_id.startswith('admin_'):
            return False
        return True
    except:
        return False

@admin_bp.route('/login', methods=['POST'])
def admin_login():
    """
    管理员登录接口
    ---
    请求参数:
        - username: 管理员用户名，字符串，必填
        - password: 管理员密码，字符串，必填
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 管理员信息和token
    """
    try:
        # 验证请求数据
        schema = AdminLoginSchema()
        data = schema.load(request.get_json())
        
        # 验证管理员账号密码
        if data['username'] != ADMIN_USERNAME or data['password'] != ADMIN_PASSWORD:
            return jsonify({
                'code': 1,
                'message': '管理员账号或密码错误',
                'data': None
            }), 401
            
        # 生成管理员JWT令牌
        access_token = create_access_token(identity=f"admin_{ADMIN_USERNAME}")
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '管理员登录成功',
            'data': {
                'admin': {
                    'username': ADMIN_USERNAME,
                    'role': 'admin',
                    'login_time': datetime.utcnow().isoformat()
                },
                'token': access_token
            }
        })
        
    except ValidationError as e:
        return jsonify({
            'code': 1,
            'message': '参数验证失败',
            'data': e.messages
        }), 400
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'登录失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/dashboard', methods=['GET'])
@jwt_required()
def get_dashboard():
    """
    获取管理后台仪表板数据
    ---
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 仪表板统计数据
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        # 统计数据
        total_users = User.query.count()
        total_notes = Note.query.count()
        users_with_brainwave = User.query.filter(User.brain_wave_file.isnot(None)).count()
        
        # 最近注册用户
        recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
        
        # 最近创建的笔记
        recent_notes = Note.query.order_by(Note.created_at.desc()).limit(5).all()
        
        return jsonify({
            'code': 0,
            'message': '获取仪表板数据成功',
            'data': {
                'statistics': {
                    'total_users': total_users,
                    'total_notes': total_notes,
                    'users_with_brainwave': users_with_brainwave,
                    'brainwave_usage_rate': round((users_with_brainwave / total_users * 100) if total_users > 0 else 0, 2)
                },
                'recent_users': [user.to_dict() for user in recent_users],
                'recent_notes': [note.to_dict() for note in recent_notes]
            }
        })
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取仪表板数据失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """
    获取用户列表
    ---
    请求参数:
        - page: 页码，默认1
        - per_page: 每页数量，默认10
        - search: 搜索关键词（可选）
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 用户列表和分页信息
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)  # 限制最大100条
        search = request.args.get('search', '', type=str)
        
        # 构建查询
        query = User.query
        if search:
            query = query.filter(
                db.or_(
                    User.username.contains(search),
                    User.email.contains(search)
                )
            )
        
        # 分页查询
        pagination = query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'code': 0,
            'message': '获取用户列表成功',
            'data': {
                'users': [user.to_dict() for user in pagination.items],
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total': pagination.total,
                    'pages': pagination.pages,
                    'has_prev': pagination.has_prev,
                    'has_next': pagination.has_next
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取用户列表失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/users/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user_detail(user_id):
    """
    获取用户详细信息
    ---
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 用户详细信息
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
        
        # 获取用户的笔记数量
        note_count = Note.query.filter_by(user_id=user_id).count()
        
        user_data = user.to_dict()
        user_data['note_count'] = note_count
        
        return jsonify({
            'code': 0,
            'message': '获取用户详情成功',
            'data': {
                'user': user_data
            }
        })
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取用户详情失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    """
    更新用户信息
    ---
    请求参数:
        - username: 用户名（可选）
        - email: 邮箱（可选）
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 更新后的用户信息
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
        
        # 验证请求数据
        schema = UpdateUserSchema()
        data = schema.load(request.get_json())
        
        # 更新用户信息
        if 'username' in data:
            # 检查用户名是否已被其他用户使用
            existing_user = User.query.filter_by(username=data['username']).first()
            if existing_user and existing_user.id != user_id:
                return jsonify({
                    'code': 1,
                    'message': '该用户名已被使用',
                    'data': None
                }), 400
            user.username = data['username']
        
        if 'email' in data:
            # 检查邮箱是否已被其他用户使用
            if data['email']:
                existing_email = User.query.filter_by(email=data['email']).first()
                if existing_email and existing_email.id != user_id:
                    return jsonify({
                        'code': 1,
                        'message': '该邮箱已被使用',
                        'data': None
                    }), 400
            user.email = data['email']
        
        db.session.commit()
        
        return jsonify({
            'code': 0,
            'message': '用户信息更新成功',
            'data': {
                'user': user.to_dict()
            }
        })
        
    except ValidationError as e:
        return jsonify({
            'code': 1,
            'message': '参数验证失败',
            'data': e.messages
        }), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'更新用户信息失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """
    删除用户
    ---
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 操作结果
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
        
        # 删除用户相关的笔记
        Note.query.filter_by(user_id=user_id).delete()
        
        # 删除用户
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({
            'code': 0,
            'message': '用户删除成功',
            'data': {
                'deleted_user_id': user_id
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'删除用户失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/notes', methods=['GET'])
@jwt_required()
def get_notes():
    """
    获取笔记列表
    ---
    请求参数:
        - page: 页码，默认1
        - per_page: 每页数量，默认10
        - user_id: 用户ID过滤（可选）
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 笔记列表和分页信息
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        user_id = request.args.get('user_id', type=int)
        
        # 构建查询
        query = Note.query
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        # 分页查询
        pagination = query.order_by(Note.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # 获取笔记并添加用户信息
        notes_data = []
        for note in pagination.items:
            note_dict = note.to_dict()
            user = User.query.get(note.user_id)
            note_dict['user'] = {
                'id': user.id,
                'username': user.username
            } if user else None
            notes_data.append(note_dict)
        
        return jsonify({
            'code': 0,
            'message': '获取笔记列表成功',
            'data': {
                'notes': notes_data,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total': pagination.total,
                    'pages': pagination.pages,
                    'has_prev': pagination.has_prev,
                    'has_next': pagination.has_next
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取笔记列表失败：{str(e)}',
            'data': None
        }), 500

@admin_bp.route('/notes/<int:note_id>', methods=['DELETE'])
@jwt_required()
def delete_note(note_id):
    """
    删除笔记
    ---
    返回响应:
        - code: 状态码，0表示成功
        - message: 提示信息
        - data: 操作结果
    """
    try:
        # 验证管理员权限
        if not verify_admin_token():
            return jsonify({
                'code': 1,
                'message': '无管理员权限',
                'data': None
            }), 403
        
        note = Note.query.get(note_id)
        if not note:
            return jsonify({
                'code': 1,
                'message': '笔记不存在',
                'data': None
            }), 404
        
        # 删除笔记
        db.session.delete(note)
        db.session.commit()
        
        return jsonify({
            'code': 0,
            'message': '笔记删除成功',
            'data': {
                'deleted_note_id': note_id
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'删除笔记失败：{str(e)}',
            'data': None
        }), 500 