from flask import Flask, request, jsonify
import sys
import os
import csv
# curl命令
# 单用户推荐
# curl -X POST http://127.0.0.1:5000/recommend/user_123 -H "Content-Type: application/json" -d "{\"user_history\": [[1, 1], [3, 1], [5, 0], [7, 1]], \"k\": 5}"

# 多用户批量推荐
# curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d "{\"user_item\": {\"a\": [[1, 1], [3, 1], [5, 0], [7, 1]], \"b\": [[1, 0], [16, 1], [6, 0], [9, 1]]}, \"k\": 5}"

# 添加当前目录到Python路径，以便导入inference_only模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_only import API_rec

api = Flask(__name__)

# 全局变量存储课程信息
course_info = {}

# 加载课程信息info，包含课程ID和课程名称，暂时没有URL
def load_course_info():
    global course_info
    try:
        with open('DRL-code-pytorch-main/Course_DQN/class_index.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:  # 确保有ID和名称
                    class_id = int(row[0])
                    class_name = row[1]
                    course_info[class_id] = class_name
        print(f"成功加载 {len(course_info)} 个课程信息")
    except Exception as e:
        print(f"加载课程信息失败: {str(e)}")

# 包含课程名称的推荐result
def format_recommendations_with_names(recommendations):
    formatted_recommendations = {}
    
    for user_id, course_ids in recommendations.items():
        user_recommendations = []
        for course_id in course_ids:
            course_name = course_info.get(course_id, f"未知课程_{course_id}")
            user_recommendations.append({
                "course_id": course_id,
                "course_name": course_name
            })
        formatted_recommendations[user_id] = user_recommendations
    
    return formatted_recommendations

# 启动时加载课程信息
load_course_info()

@api.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    推荐API端点
    请求格式:
    POST /recommend
    {
        "user_item": {
            "user_a": [[1, 1], [3, 1], [5, 0], [7, 1]],
            "user_b": [[1, 0], [16, 1], [6, 0], [9, 1]]
        },
        "k": 10
    }
    """
    try:
        # 获取请求数据
        data = request.json
        
        if not data or 'user_item' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少user_item参数'
            }), 400
        
        user_item = data['user_item']
        k = data.get('k', 10)  # 默认推荐10个
        model_path = data.get('model_path', 'DRL-code-pytorch-main/Course_DQN/saved_models/direct_rec_model')
        
        # 转换数据格式：将列表转换为元组
        formatted_user_item = {}
        for user_id, history in user_item.items():
            formatted_user_item[user_id] = [(item[0], item[1]) for item in history]
        
        # 调用推荐算法
        recommendations = API_rec(formatted_user_item, model_path, k)
        
        # 格式化推荐结果，添加课程名称
        formatted_recommendations = format_recommendations_with_names(recommendations)
        
        return jsonify({
            'status': 'success',
            'recommendations': formatted_recommendations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'推荐失败: {str(e)}'
        }), 500

@api.route('/recommend/<user_id>', methods=['POST'])
def get_user_recommendation(user_id):
    """
    单用户推荐API端点
    请求格式:
    POST /recommend/user_123
    {
        "user_history": [[1, 1], [3, 1], [5, 0], [7, 1]],
        "k": 10
    }
    """
    try:
        data = request.json
        
        if not data or 'user_history' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少user_history参数'
            }), 400
        
        user_history = data['user_history']
        k = data.get('k', 10)
        model_path = data.get('model_path', 'DRL-code-pytorch-main/Course_DQN/saved_models/direct_rec_model')
        
        # 格式化单用户数据
        user_item = {user_id: [(item[0], item[1]) for item in user_history]}
        
        # 调用推荐算法
        recommendations = API_rec(user_item, model_path, k)
        
        # 格式化推荐结果，添加课程名称
        formatted_recommendations = format_recommendations_with_names(recommendations)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'recommendations': formatted_recommendations.get(user_id, [])
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'推荐失败: {str(e)}'
        }), 500

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Course Recommendation API',
        'loaded_courses': len(course_info)
    })
# API文档
@api.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '课程推荐API',
        'endpoints': {
            'POST /recommend': '批量用户推荐',
            'POST /recommend/<user_id>': '单用户推荐',
            'GET /health': '健康检查'
        },
        'example_request': {
            'url': '/recommend',
            'method': 'POST',
            'body': {
                'user_item': {
                    'user_a': [[1, 1], [3, 1], [5, 0]],
                    'user_b': [[2, 1], [4, 0], [6, 1]]
                },
                'k': 10
            }
        },
        'loaded_courses': len(course_info)
    })

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=5000)
