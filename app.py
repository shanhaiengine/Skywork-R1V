from flask import Flask, request, jsonify
import torch
from transformers import AutoModel, AutoTokenizer
from utils import load_image, split_model

app = Flask(__name__)

# 全局变量存储模型和tokenizer
model = None
tokenizer = None

def load_model(model_path):
    global model, tokenizer
    if model is None:
        device_map = split_model(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        model_path = request.json.get('model_path')
        if not model_path:
            return jsonify({'error': '需要提供model_path参数'}), 400
        
        load_model(model_path)
        return jsonify({'message': '模型加载成功'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/inference', methods=['POST'])
def inference():
    try:
        if model is None or tokenizer is None:
            return jsonify({'error': '模型未初始化，请先调用initialize接口'}), 400

        data = request.json
        image_paths = data.get('image_paths', [])
        question = data.get('question')

        if not image_paths or not question:
            return jsonify({'error': '需要提供image_paths和question参数'}), 400

        # 处理图片
        pixel_values = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda() 
                       for img_path in image_paths]
        
        if len(pixel_values) > 1:
            num_patches_list = [img.size(0) for img in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = pixel_values[0]
            num_patches_list = None

        prompt = "<image>\n" * len(image_paths) + question
        generation_config = dict(
            max_new_tokens=64000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05
        )

        response = model.chat(tokenizer, pixel_values, prompt, 
                            generation_config, num_patches_list=num_patches_list)

        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 