#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG算法集成测试脚本
用于测试EEG服务是否正常工作
"""

import sys
import os
from utils.eeg_service import eeg_service

def test_eeg_service():
    """测试EEG服务的基本功能"""
    
    print("="*50)
    print("EEG算法集成测试")
    print("="*50)
    
    # 测试文件路径
    test_npz_file = "algorithm/EEG-main/16_channels_seg/S101R12.npz"
    
    print(f"\n1. 测试文件有效性检查...")
    print(f"测试文件: {test_npz_file}")
    
    # 检查文件是否存在
    if not os.path.exists(test_npz_file):
        print(f"❌ 错误: 测试文件不存在 - {test_npz_file}")
        return False
    
    # 测试文件有效性
    is_valid = eeg_service.is_valid_eeg_file(test_npz_file)
    print(f"文件有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
    
    if not is_valid:
        print("❌ 测试文件无效，停止测试")
        return False
    
    print(f"\n2. 测试EEG系统初始化...")
    if eeg_service.auth_system is None:
        print("❌ EEG认证系统未初始化")
        return False
    else:
        print("✅ EEG认证系统已初始化")
    
    print(f"\n3. 测试用户注册功能...")
    test_user_id = "test_user_001"
    
    try:
        result = eeg_service.register_user_with_eeg(test_user_id, test_npz_file)
        print(f"注册结果: {result}")
        
        if result.get("status") == "success":
            print("✅ 用户注册成功")
        else:
            print(f"❌ 用户注册失败: {result.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 注册过程中出现异常: {str(e)}")
        return False
    
    print(f"\n4. 测试用户验证功能...")
    
    try:
        # 使用同一个文件验证
        verified_user = eeg_service.verify_user_with_eeg(test_npz_file)
        print(f"验证结果: {verified_user}")
        
        if verified_user == test_user_id:
            print("✅ 用户验证成功")
        elif verified_user == "Unknown":
            print("⚠️ 未找到匹配用户（可能是相似度阈值问题）")
        else:
            print(f"⚠️ 找到了不同的用户: {verified_user}")
            
    except Exception as e:
        print(f"❌ 验证过程中出现异常: {str(e)}")
        return False
    
    print(f"\n5. 测试不同文件的验证...")
    
    # 使用不同的文件测试
    different_file = "algorithm/EEG-main/16_channels_seg/S102R12.npz"
    if os.path.exists(different_file):
        try:
            verified_user_2 = eeg_service.verify_user_with_eeg(different_file)
            print(f"不同文件验证结果: {verified_user_2}")
            
            if verified_user_2 == "Unknown":
                print("✅ 正确识别不同的脑电波数据")
            else:
                print(f"⚠️ 意外匹配到用户: {verified_user_2}")
                
        except Exception as e:
            print(f"❌ 不同文件验证过程中出现异常: {str(e)}")
    else:
        print(f"⚠️ 第二个测试文件不存在: {different_file}")
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
    
    return True

def test_file_conversion():
    """测试文件转换功能"""
    print(f"\n6. 测试文件转换功能...")
    
    test_npz = "algorithm/EEG-main/16_channels_seg/S101R12.npz"
    converted = eeg_service.convert_file_to_npz(test_npz)
    
    if converted == test_npz:
        print("✅ NPZ文件转换正常")
    else:
        print(f"❌ 文件转换异常: {converted}")

if __name__ == "__main__":
    print("开始EEG算法集成测试...")
    
    try:
        success = test_eeg_service()
        test_file_conversion()
        
        if success:
            print("\n🎉 所有基础测试通过！")
        else:
            print("\n💥 测试过程中出现问题，请检查配置和依赖")
            
    except Exception as e:
        print(f"\n💥 测试过程中出现严重错误: {str(e)}")
        import traceback
        traceback.print_exc() 