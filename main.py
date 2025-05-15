from flask import Flask, request, jsonify
from reciept import receipt_model,save_receipt_in_mongodb
from budget import parse_budget,save_in_db
from recommender import financial_recommender
from datetime import datetime

app = Flask(__name__)

@app.route('/parse-receipt', methods=['POST'])
def parse_reciept():
    data = request.json
    image_url = data.get('image_url')
    user_id = data.get('user_id')
    category = data.get('category')
    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400
    try:
        llm_response = receipt_model(image_url)
        print(llm_response)
        save_receipt_in_mongodb(user_id=user_id,llm_response=llm_response,date=datetime.now().strftime("%Y-%m-%d"),category=category)
        return jsonify({"message": "Receipt parsed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/generate-budget',methods=['POST'])
def generate_budget():
    data = request.json
    user_id = data.get('user_id')
    description = data.get('description')
    try:
        response = parse_budget(description)
        save_in_db(user_id,response)
        return jsonify({"message": "Budget generated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get-recommendations',methods=['POST'])
def recommendations():
    data = request.json
    user_id = data.get('user_id')
    try:
        recommendations = financial_recommender(user_id)
        return jsonify({"message": "Recommendations generated successfully","response":recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)