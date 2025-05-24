from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import User, Note, db
from marshmallow import Schema, fields, validate, ValidationError
from utils.file_utils import save_uploaded_file, allowed_file
import os

note_bp = Blueprint('note', __name__, url_prefix='/api/notes')

class NoteSchema(Schema):
    """笔记验证模式"""
    title = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    content = fields.Str(missing=None)

@note_bp.route('/', methods=['GET'])
@jwt_required()
def get_notes():
    """
    获取用户所有笔记列表
    ---
    请求参数:
        无（通过JWT令牌识别用户）
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 笔记列表，每条笔记包含id、标题和背景图片
    """
    try:
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 获取该用户的所有笔记
        notes = Note.query.filter_by(user_id=user.id).order_by(Note.updated_at.desc()).all()
        
        # 返回笔记摘要列表（只包含id、标题和背景图）
        note_list = [note.to_summary_dict() for note in notes]
        
        return jsonify({
            'code': 0,
            'message': '获取笔记列表成功',
            'data': {
                'notes': note_list
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取笔记列表失败：{str(e)}',
            'data': None
        }), 500

@note_bp.route('/<int:note_id>', methods=['GET'])
@jwt_required()
def get_note(note_id):
    """
    获取指定笔记详情
    ---
    请求参数:
        - note_id: 笔记ID，整数，必填
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 笔记详情，包含完整内容
    """
    try:
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 获取指定笔记
        note = Note.query.filter_by(id=note_id, user_id=user.id).first()
        
        if not note:
            return jsonify({
                'code': 1,
                'message': '笔记不存在或无权访问',
                'data': None
            }), 404
        
        # 返回笔记详情
        return jsonify({
            'code': 0,
            'message': '获取笔记详情成功',
            'data': {
                'note': note.to_dict()
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': f'获取笔记详情失败：{str(e)}',
            'data': None
        }), 500

@note_bp.route('/', methods=['POST'])
@jwt_required()
def create_note():
    """
    创建新笔记
    ---
    请求参数:
        - title: 标题，字符串，必填，最大长度200
        - content: 内容，字符串，选填
        - background_image: 背景图片，文件，选填，支持jpg、jpeg、png、gif格式
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 创建的笔记详情
    """
    try:
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 处理表单数据
        form_data = {}
        if request.form.get('title'):
            form_data['title'] = request.form.get('title')
        if request.form.get('content'):
            form_data['content'] = request.form.get('content')
            
        # 验证表单数据
        schema = NoteSchema()
        data = schema.load(form_data)
        
        # 处理背景图片
        background_image_path = None
        if 'background_image' in request.files:
            bg_image = request.files['background_image']
            if bg_image.filename != '':
                bg_image_path = save_uploaded_file(bg_image, subfolder='note_backgrounds', file_type='avatar')
                if not bg_image_path:
                    return jsonify({
                        'code': 1,
                        'message': '背景图片格式不支持，请上传jpg、jpeg、png或gif格式的图片',
                        'data': None
                    }), 400
                background_image_path = bg_image_path
        
        # 创建新笔记
        new_note = Note(
            user_id=user.id,
            title=data['title'],
            content=data.get('content'),
            background_image=background_image_path
        )
        
        db.session.add(new_note)
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '创建笔记成功',
            'data': {
                'note': new_note.to_dict()
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
            'message': f'创建笔记失败：{str(e)}',
            'data': None
        }), 500

@note_bp.route('/<int:note_id>', methods=['PUT'])
@jwt_required()
def update_note(note_id):
    """
    更新笔记
    ---
    请求参数:
        - note_id: 笔记ID，整数，必填
        - title: 标题，字符串，选填，最大长度200
        - content: 内容，字符串，选填
        - background_image: 背景图片，文件，选填，支持jpg、jpeg、png、gif格式
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 更新后的笔记详情
    """
    try:
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 获取要更新的笔记
        note = Note.query.filter_by(id=note_id, user_id=user.id).first()
        
        if not note:
            return jsonify({
                'code': 1,
                'message': '笔记不存在或无权访问',
                'data': None
            }), 404
            
        # 处理表单数据
        form_data = {}
        if request.form.get('title'):
            form_data['title'] = request.form.get('title')
        if request.form.get('content') is not None:  # 允许内容为空字符串
            form_data['content'] = request.form.get('content')
            
        # 验证表单数据
        schema = NoteSchema(partial=True)  # partial=True表示允许部分字段更新
        data = schema.load(form_data)
        
        # 处理背景图片
        if 'background_image' in request.files:
            bg_image = request.files['background_image']
            if bg_image.filename != '':
                # 删除旧背景图（如果存在）
                if note.background_image and os.path.exists(os.path.join(os.getcwd(), note.background_image)):
                    try:
                        os.remove(os.path.join(os.getcwd(), note.background_image))
                    except:
                        # 忽略删除失败的错误
                        pass
                
                # 保存新背景图
                bg_image_path = save_uploaded_file(bg_image, subfolder='note_backgrounds', file_type='avatar')
                if not bg_image_path:
                    return jsonify({
                        'code': 1,
                        'message': '背景图片格式不支持，请上传jpg、jpeg、png或gif格式的图片',
                        'data': None
                    }), 400
                note.background_image = bg_image_path
        
        # 更新笔记信息
        if 'title' in data:
            note.title = data['title']
        if 'content' in data:
            note.content = data['content']
        
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '更新笔记成功',
            'data': {
                'note': note.to_dict()
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
            'message': f'更新笔记失败：{str(e)}',
            'data': None
        }), 500

@note_bp.route('/<int:note_id>', methods=['DELETE'])
@jwt_required()
def delete_note(note_id):
    """
    删除笔记
    ---
    请求参数:
        - note_id: 笔记ID，整数，必填
    返回响应:
        - code: 状态码，0表示成功，非0表示失败
        - message: 提示信息
        - data: 操作结果
    """
    try:
        # 获取当前登录用户
        current_user_id = get_jwt_identity()
        user = User.query.get(int(current_user_id))
        
        if not user:
            return jsonify({
                'code': 1,
                'message': '用户不存在',
                'data': None
            }), 404
            
        # 获取要删除的笔记
        note = Note.query.filter_by(id=note_id, user_id=user.id).first()
        
        if not note:
            return jsonify({
                'code': 1,
                'message': '笔记不存在或无权访问',
                'data': None
            }), 404
            
        # 删除背景图片（如果存在）
        if note.background_image and os.path.exists(os.path.join(os.getcwd(), note.background_image)):
            try:
                os.remove(os.path.join(os.getcwd(), note.background_image))
            except:
                # 忽略删除失败的错误
                pass
        
        # 删除笔记
        db.session.delete(note)
        db.session.commit()
        
        # 返回成功响应
        return jsonify({
            'code': 0,
            'message': '删除笔记成功',
            'data': {
                'success': True
            }
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 2,
            'message': f'删除笔记失败：{str(e)}',
            'data': None
        }), 500 