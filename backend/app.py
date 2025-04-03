from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Tạo thư mục này trước

# Load mô hình Flan-T5
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Tiền xử lý câu hỏi
def preprocess_input(user_input, file_content=None):
    instruction = (
        "Please explain in detail and provide multiple real-world examples. "
        "Make sure the answer is at least 200 words long, clear, and easy to understand. "
    )
    if file_content:
        instruction += f"\nHere is some file content to consider:\n{file_content}\n"
    return instruction + "\nHere is the question: " + user_input


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = preprocess_input(data['text'])

    # Tokenize và sinh câu trả lời
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=1024,
        temperature=0.7,
        num_beams=5,
        top_k=100,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'answer': output_text})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Missing file or text'}), 400

    file = request.files['file']
    question = request.form['text']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Đọc nội dung file
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    input_text = preprocess_input(question, file_content)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=1024,
        temperature=0.7,
        num_beams=5,
        top_k=100,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'answer': output_text})


if __name__ == '__main__':
    app.run(debug=True)



# Đoạn văn bản mới cần tóm tắt
new_input_text = "The large, colorful bird flew across the sky and landed on a tree branch. It looked around for food and then took off again to find a mate."
# Tokenize đầu vào mới
new_inputs = tokenizer(new_input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
# Dự đoán tóm tắt
summary_ids = model.generate(new_inputs['input_ids'], max_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)
# Giải mã kết quả
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary_text)

