from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from models import User, db
from marshmallow import Schema, fields, validate, ValidationError
from utils.file_utils import save_uploaded_file, allowed_file
import os

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# 创建一个令牌黑名单集合
# 在实际生产环境中，应该使用Redis等缓存存储
jwt_blocklist = set()

class RegisterSchema(Schema):
    """注册请求验证模式"""
    account = fields.Str(required=True, validate=validate.Length(min=5, max=30))
    password = fields.Str(required=True, validate=validate.Length(min=6, max=50))

class LoginSchema(Schema):
    """登录请求验证模式"""
    account = fields.Str(required=True)
    password = fields.Str(required=True)

class ChangePasswordSchema(Schema):
    """修改密码请求验证模式"""
    old_password = fields.Str(required=True)
    new_password = fields.Str(required=True, validate=validate.Length(min=6, max=50))

class UpdateProfileSchema(Schema):
    """更新个人信息请求验证模式"""
    username = fields.Str(required=False, validate=validate.Length(min=2, max=30))
    email = fields.Email(required=False, allow_none=True)

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    用户注册接口
    ---
    请求参数:
        - account: 账号，字符串，必填，长度5-30
        - password: 密码，字符串，必填，长度6-50
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 用户信息，包括token
    """
    try:
        # 验证请求数据
        schema = RegisterSchema()
        data = schema.load(request.get_json())
        
        # 检查账号是否已存在
        if User.query.filter_by(username=data['account']).first():
            return jsonify({
                'code': 1,
                'message': '该账号已被注册',
                'data': None
            }), 400
            
        # 生成用户名，格式为Userxxx
        username = User.generate_username()
        
        # 创建新用户
        new_user = User(
            username=data['account'],  # 用account作为username
            password=data['password'],
            email=None,
            avatar=None
        )
        
        # 保存到数据库
        db.session.add(new_user)
        db.session.commit()
        
        # 生成JWT令牌（确保身份标识是字符串类型）
        access_token = create_access_token(identity=str(new_user.id))
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '注册成功，已自动登录',
            'data': {
                'user': new_user.to_dict(),
                'token': access_token,
                'is_logged_in': True
            }
        }), 200
        
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
            'message': f'注册失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    用户登录接口
    ---
    请求参数:
        - account: 账号，字符串，必填
        - password: 密码，字符串，必填
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 用户信息，包括token
    """
    try:
        # 验证请求数据
        schema = LoginSchema()
        data = schema.load(request.get_json())
        
        # 查找用户
        user = User.query.filter_by(username=data['account']).first()
        
        # 验证用户和密码
        if not user or not user.check_password(data['password']):
            return jsonify({
                'code': 1,
                'message': '账号或密码错误',
                'data': None
            }), 401
            
        # 生成JWT令牌（确保身份标识是字符串类型）
        access_token = create_access_token(identity=str(user.id))
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '登录成功',
            'data': {
                'user': user.to_dict(),
                'token': access_token,
                'is_logged_in': True
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

@auth_bp.route('/brain-wave-login', methods=['POST'])
def brain_wave_login():
    """
    用户脑电波文件登录接口（如果该脑电波数据未注册过，则自动注册新用户）
    ---
    请求参数:
        - brain_wave_file: 脑电波文件，必填，文件类型，通过form-data方式提交
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 用户信息，包括token和登录状态
        - is_new_user: 是否为新注册的用户
        - password: 如果是新用户，返回默认密码(123456)
    """
    try:
        # 检查请求中是否包含文件
        if 'brain_wave_file' not in request.files:
            return jsonify({
                'code': 1,
                'message': '请上传脑电波文件',
                'data': None
            }), 400
            
        file = request.files['brain_wave_file']
        
        # 检查文件是否有效
        if file.filename == '':
            return jsonify({
                'code': 1,
                'message': '未选择文件',
                'data': None
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'code': 1,
                'message': '不支持的文件类型，请上传 NPZ、EEG、DAT、TXT 或 BIN 格式的文件',
                'data': None
            }), 400
            
        # 保存文件
        file_path = save_uploaded_file(file, subfolder='brain_wave')
        if not file_path:
            return jsonify({
                'code': 1,
                'message': '文件保存失败',
                'data': None
            }), 500
            
        # 使用EEG算法查询用户
        user = User.find_by_brain_wave_file(file_path)
        
        # 如果未找到用户，则创建新用户
        is_new_user = False
        
        if not user:
            # 创建新用户（使用EEG算法注册）
            result = User.create_with_brain_wave(file_path)
            if not result:
                return jsonify({
                    'code': 1,
                    'message': '脑电波数据处理失败，无法创建用户',
                    'data': None
                }), 500
                
            user, default_password = result
            db.session.add(user)
            db.session.commit()
            is_new_user = True
            
        # 生成JWT令牌（确保身份标识是字符串类型）
        access_token = create_access_token(identity=str(user.id))
        
        # 返回成功响应
        response_data = {
            'code': 0,
            'message': '脑电波登录成功',
            'data': {
                'user': user.to_dict(),
                'token': access_token,
                'is_logged_in': True
            },
            'is_new_user': is_new_user
        }
        
        # 如果是新用户，在消息中说明默认密码
        if is_new_user:
            response_data['data']['default_password'] = '123456'
            response_data['message'] = '脑电波登录成功，已自动创建新账号（默认密码：123456）'
        
        return jsonify(response_data)
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'脑电波登录失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/update-brain-wave', methods=['POST'])
@jwt_required()
def update_brain_wave():
    """
    更新用户脑电波文件接口
    ---
    请求参数:
        - brain_wave_file: 脑电波文件，必填，文件类型，通过form-data方式提交
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 更新后的用户信息
    """
    try:
        # 获取当前登录用户ID
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 检查请求中是否包含文件
        if 'brain_wave_file' not in request.files:
            return jsonify({
                'code': 1,
                'message': '请上传脑电波文件',
                'data': None
            }), 400
            
        file = request.files['brain_wave_file']
        
        # 检查文件是否有效
        if file.filename == '':
            return jsonify({
                'code': 1,
                'message': '未选择文件',
                'data': None
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'code': 1,
                'message': '不支持的文件类型，请上传 NPZ、EEG、DAT、TXT 或 BIN 格式的文件',
                'data': None
            }), 400
            
        # 删除旧文件（如果存在）
        if user.brain_wave_file and os.path.exists(os.path.join(os.getcwd(), user.brain_wave_file)):
            try:
                os.remove(os.path.join(os.getcwd(), user.brain_wave_file))
            except:
                # 忽略删除失败的错误
                print(f"删除脑电波文件失败：{os.path.join(os.getcwd(), user.brain_wave_file)}")
            
        # 保存新文件
        file_path = save_uploaded_file(file, subfolder='brain_wave')
        if not file_path:
            return jsonify({
                'code': 1,
                'message': '文件保存失败',
                'data': None
            }), 500
            
        # 使用EEG算法更新用户脑电波文件
        if not user.update_brain_wave_file(file_path):
            return jsonify({
                'code': 1,
                'message': '脑电波数据处理失败，文件更新失败',
                'data': None
            }), 500
            
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '脑电波文件更新成功',
            'data': {
                'user': user.to_dict()
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'更新脑电波文件失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/user-info', methods=['GET'])
@jwt_required()
def get_user_info():
    """
    获取用户个人信息接口
    ---
    请求参数:
        无（通过JWT令牌识别用户）
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 用户信息
    """
    try:
        # 获取当前登录用户ID
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '获取用户信息成功',
            'data': {
                'user': user.to_dict()
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取用户信息失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """
    修改密码接口
    ---
    请求参数:
        - old_password: 原密码，字符串，必填
        - new_password: 新密码，字符串，必填，长度6-50
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 操作结果
    """
    try:
        # 验证请求数据
        schema = ChangePasswordSchema()
        data = schema.load(request.get_json())
        
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
        
        # 验证原密码是否正确
        if not user.check_password(data['old_password']):
            return jsonify({
                'code': 1,
                'message': '原密码错误',
                'data': None
            }), 400
        
        # 设置新密码
        user.set_password(data['new_password'])
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '密码修改成功',
            'data': {
                'success': True
            }
        }), 200
        
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
            'message': f'修改密码失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/update-profile', methods=['POST'])
@jwt_required()
def update_profile():
    """
    修改个人信息接口
    ---
    请求参数:
        - username: 用户名，字符串，选填，长度2-30
        - email: 邮箱，字符串，选填，符合邮箱格式
        - avatar: 头像文件，文件类型，选填，支持jpg、jpeg、png、gif格式
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 更新后的用户信息
    """
    try:
        print("\033[32m=== 开始处理更新个人信息请求 ===\033[0m")
        print("\033[32m请求表单数据：{}\033[0m".format(request.form))
        print("\033[32m请求文件：{}\033[0m".format(request.files))
        
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        print("\033[32m当前用户ID：{}\033[0m".format(current_user_id))
        print("\033[32m当前用户信息：{}\033[0m".format(user.to_dict() if user else None))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404

        # 处理表单数据
        form_data = {}
        print("\033[32m请求表单数据：{}\033[0m".format(request.form))
        if request.form.get('username'):
            form_data['username'] = request.form.get('username')
        if request.form.get('email'):
            form_data['email'] = request.form.get('email')

        print("\033[32m处理后的表单数据：{}\033[0m".format(form_data))

        # 验证表单数据
        schema = UpdateProfileSchema()
        data = schema.load(form_data)
        print("\033[32mSchema验证后的数据：{}\033[0m".format(data))
        
        # 检查用户名是否已被其他用户使用（如果提供了用户名）
        if data.get('username'):
            print("\033[32m检查用户名是否已被使用：{}\033[0m".format(data['username']))
            existing_user = User.query.filter_by(username=data['username']).first()
            print("\033[32m查找到的已存在用户：{}\033[0m".format(existing_user.to_dict() if existing_user else None))
            if existing_user and existing_user.id != int(current_user_id):
                return jsonify({
                    'code': 1,
                    'message': '该用户名已被使用',
                    'data': None
                }), 400
        
        # 检查邮箱是否已被其他用户使用（如果提供了邮箱）
        if data.get('email'):
            print("\033[32m检查邮箱是否已被使用：{}\033[0m".format(data['email']))
            existing_email = User.query.filter_by(email=data['email']).first()
            print("\033[32m查找到的已存在邮箱用户：{}\033[0m".format(existing_email.to_dict() if existing_email else None))
            if existing_email and existing_email.id != int(current_user_id):
                return jsonify({
                    'code': 1,
                    'message': '该邮箱已被使用',
                    'data': None
                }), 400

        # 处理头像文件上传
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            print("\033[32m上传的头像文件名：{}\033[0m".format(avatar_file.filename))
            if avatar_file.filename != '':
                # 删除旧的头像文件（如果存在）
                if user.avatar and os.path.exists(os.path.join(os.getcwd(), user.avatar)):
                    try:
                        print("\033[32m尝试删除旧头像文件：{}\033[0m".format(user.avatar))
                        os.remove(os.path.join(os.getcwd(), user.avatar))
                        print("\033[32m旧头像文件删除成功\033[0m")
                    except Exception as e:
                        print("\033[32m删除旧头像文件失败：{}\033[0m".format(str(e)))
                        pass

                # 保存新的头像文件
                avatar_path = save_uploaded_file(avatar_file, subfolder='avatars', file_type='avatar')
                print("\033[32m新头像文件保存路径：{}\033[0m".format(avatar_path))
                if not avatar_path:
                    return jsonify({
                        'code': 1,
                        'message': '头像文件格式不支持，请上传jpg、jpeg、png或gif格式的图片',
                        'data': None
                    }), 400
                user.avatar = avatar_path
        
        # 更新用户信息（只更新提供的字段）
        print("\033[32m=== 开始更新用户信息 ===\033[0m")
        if data.get('username'):
            print("\033[32m更新用户名：{} -> {}\033[0m".format(user.username, data['username']))
            user.username = data['username']
        if 'email' in data:  # 允许设置email为None
            print("\033[32m更新邮箱：{} -> {}\033[0m".format(user.email, data['email']))
            user.email = data['email']
        
        db.session.commit()
        print("\033[32m数据库更新成功\033[0m")
        
        updated_user = user.to_dict()
        print("\033[32m更新后的用户信息：{}\033[0m".format(updated_user))
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '个人信息更新成功',
            'data': {
                'user': updated_user
            }
        }), 200
        
    except ValidationError as e:
        print("\033[32m参数验证失败：{}\033[0m".format(e.messages))
        return jsonify({
            'code': 1,
            'message': '参数验证失败',
            'data': e.messages
        }), 400
    except Exception as e:
        print("\033[32m更新个人信息失败：{}\033[0m".format(str(e)))
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'更新个人信息失败：{str(e)}',
            'data': None
        }), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    用户登出接口
    ---
    请求参数:
        无（通过JWT令牌识别用户）
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 操作结果
    """
    try:
        # 获取当前的JWT令牌
        jti = get_jwt()["jti"]
        
        # 将JWT ID添加到黑名单集合中
        jwt_blocklist.add(jti)
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '用户已成功登出',
            'data': {
                'success': True
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'登出失败：{str(e)}',
            'data': None
        }), 500 