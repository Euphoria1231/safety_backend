import os
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from models import db, User
from routes import auth_bp, note_bp, admin_bp
from routes.auth import jwt_blocklist
from config import config

def create_app(config_name='default'):
    """应用工厂函数"""
    app = Flask(__name__)
    
    # 加载配置
    app.config.from_object(config[config_name])
    
    # 初始化扩展
    db.init_app(app)
    jwt = JWTManager(app)
    
    # 配置CORS以支持跨域请求
    CORS(app, supports_credentials=True, origins=['*'])
    
    # 配置JWT回调，检查令牌是否在黑名单中
    @jwt.token_in_blocklist_loader
    def check_if_token_in_blocklist(jwt_header, jwt_payload):
        # 检查JWT ID是否在黑名单集合中
        jti = jwt_payload["jti"]
        return jti in jwt_blocklist
    
    # 令牌被拒绝时的回调
    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'code': 401,
            'message': '用户已登出或会话已过期，请重新登录',
            'data': None
        }), 401
    
    # 注册蓝图
    app.register_blueprint(auth_bp)
    app.register_blueprint(note_bp)
    app.register_blueprint(admin_bp)
    
    # 创建数据库表（开发环境）
    with app.app_context():
        if config_name == 'development':
            db.create_all()
    
    # 基础路由
    @app.route('/')
    def index():
        return "Safety Backend API 服务正在运行"
    
    @app.route('/health')
    def health_check():
        return jsonify({"status": "ok", "version": "1.0.0"})
    
    # 错误处理
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"code": 404, "message": "接口不存在", "data": None}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"code": 500, "message": "服务器内部错误", "data": None}), 500
    
    return app

# 直接运行此文件时，使用开发环境配置
if __name__ == '__main__':
    app = create_app('development')
    # 设置host为0.0.0.0以监听所有网络接口，允许外部访问
    # 设置端口为5000（Flask默认端口）
    app.run(host='0.0.0.0', port=5000, debug=True)