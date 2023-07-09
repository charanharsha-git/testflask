from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Sample route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the AJAX request and generate the dataframe
@app.route('/generate_dataframe', methods=['POST'])
def generate_dataframe():
    store_number = int(request.json['storeNumber'])
    promotion_weightage = float(request.json['promotionWeightage'])
    forecast_days = int(request.json['forecastDays'])
    
    # Your logic to generate the dataframe based on the inputs
    # Replace the following example code with your actual implementation
    
    # Generate example dataframe with store number and dates
    dates = pd.date_range(start='1/1/2023', periods=forecast_days)
    df = pd.DataFrame({'Date': dates})
    df['StoreNumber'] = store_number
    
    # Add example columns with random data
    df['Column1'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    df['Column2'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540]
    
    # Convert the dataframe to JSON and return
    data = df.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
