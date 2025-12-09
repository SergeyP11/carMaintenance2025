import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, render_template, request, jsonify
import os


# Step 1: Generate synthetic dataset for car maintenance costs
def generate_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        car_type = random.randint(1, 4)  # 1: Sedan, 2: SUV, 3: Truck, 4: Luxury
        age = random.uniform(1, 20)  # Years
        mileage = random.uniform(10000, 200000)  # KM
        engine_type = random.randint(1, 3)  # 1: Gas, 2: Diesel, 3: Electric
        service_level = random.randint(1, 3)  # 1: Basic, 2: Standard, 3: Premium
        parts_quality = random.randint(1, 3)  # 1: Economy, 2: Standard, 3: High-end

        # Empirical cost calculation
        base_cost = 500 + (age * 50) + (mileage / 1000) * 10
        if car_type == 4: base_cost *= 1.5
        if engine_type == 3: base_cost *= 1.2
        base_cost += service_level * 200
        base_cost += parts_quality * 150
        total_cost = base_cost + random.uniform(-200, 200)  # Add noise

        data.append({
            'car_type': car_type,
            'age': age,
            'mileage': mileage,
            'engine_type': engine_type,
            'service_level': service_level,
            'parts_quality': parts_quality,
            'total_cost': total_cost
        })

    df = pd.DataFrame(data)
    df.to_csv('car_maintenance_data.csv', index=False)
    print(f"Generated dataset with {num_samples} samples.")


# Step 2: Train the model
def train_model():
    data = pd.read_csv('car_maintenance_data.csv')
    X = data.drop('total_cost', axis=1)
    y = data['total_cost']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")

    joblib.dump(model, 'maintenance_cost_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("Model trained and saved.")


# Step 3: Class wrapper for the model
class CarMaintenanceSystem:
    def __init__(self):
        if not os.path.exists('car_maintenance_data.csv'):
            generate_dataset()
        if not os.path.exists('maintenance_cost_model.pkl') or not os.path.exists('feature_scaler.pkl'):
            train_model()
        self.model = joblib.load('maintenance_cost_model.pkl')
        self.scaler = joblib.load('feature_scaler.pkl')

    def predict_cost(self, features):
        try:
            features_scaled = self.scaler.transform([features])
            predicted_cost = self.model.predict(features_scaled)[0]
            return predicted_cost
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise

    def optimize_for_budget(self, budget, n_variants=5):
        variants = []
        for _ in range(100):
            car = self._generate_random_car()
            features = self._car_to_features(car)
            cost = self.predict_cost(features)
            utility = self._calculate_utility(car, cost, budget)
            if cost <= budget * 1.1:
                variants.append((car, cost, utility))

        variants.sort(key=lambda x: x[2], reverse=True)
        return variants[:n_variants]

    def _generate_random_car(self):
        return {
            'car_type': random.randint(1, 4),
            'age': random.uniform(1, 20),
            'mileage': random.uniform(10000, 200000),
            'engine_type': random.randint(1, 3),
            'service_level': random.randint(1, 3),
            'parts_quality': random.randint(1, 3)
        }

    def _car_to_features(self, car):
        return [
            car['car_type'],
            car['age'],
            car['mileage'],
            car['engine_type'],
            car['service_level'],
            car['parts_quality']
        ]

    def _calculate_utility(self, car, cost, budget):
        # Utility based on mileage (lower mileage is better) and cost closeness to budget
        mileage_penalty = car['mileage'] / 100000
        if cost > budget:
            penalty = (cost - budget) / budget
            return - (1 / mileage_penalty) * (1 + penalty)
        return (1 / mileage_penalty) * (1 - abs(cost - budget) / budget)


# Step 4: Flask web application
app = Flask(__name__)

# Dictionaries for displaying parameter names
CAR_TYPES = {1: 'Sedan', 2: 'SUV', 3: 'Truck', 4: 'Luxury'}
ENGINE_TYPES = {1: 'Gas', 2: 'Diesel', 3: 'Electric'}
SERVICE_LEVELS = {1: 'Basic', 2: 'Standard', 3: 'Premium'}
PARTS_QUALITIES = {1: 'Economy', 2: 'Standard', 3: 'High-end'}

system = CarMaintenanceSystem()


@app.route('/')
def index():
    return render_template('index.html',
                           car_types=CAR_TYPES,
                           engine_types=ENGINE_TYPES,
                           service_levels=SERVICE_LEVELS,
                           parts_qualities=PARTS_QUALITIES)


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        budget = float(request.form.get('budget', 0))
        variants = system.optimize_for_budget(budget)
        results = []
        for car, cost, _ in variants:
            results.append({
                'car_type': CAR_TYPES[car['car_type']],
                'age': round(car['age'], 1),
                'mileage': round(car['mileage'], 0),
                'engine_type': ENGINE_TYPES[car['engine_type']],
                'service_level': SERVICE_LEVELS[car['service_level']],
                'parts_quality': PARTS_QUALITIES[car['parts_quality']],
                'total_cost': round(cost, 0)
            })
        return jsonify({'variants': results})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_cost', methods=['POST'])
def predict_cost():
    try:
        features = [
            int(request.form['car_type']),
            float(request.form['age']),
            float(request.form['mileage']),
            int(request.form['engine_type']),
            int(request.form['service_level']),
            int(request.form['parts_quality'])
        ]
        cost = system.predict_cost(features)
        return jsonify({
            'predicted_cost': round(cost, 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# HTML template (save as templates/index.html)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Car Maintenance Cost Calculator</title>
</head>
<body>
    <h1>Car Maintenance Cost Calculator</h1>

    <h2>Budget Mode</h2>
    <form id="budget-form">
        <label for="budget">Budget:</label>
        <input type="number" id="budget" name="budget" required>
        <button type="submit">Calculate</button>
    </form>
    <div id="budget-results"></div>

    <h2>Manual Mode</h2>
    <form id="manual-form">
        <label for="car_type">Car Type:</label>
        <select id="car_type" name="car_type">
            {% for key, value in car_types.items() %}
                <option value="{{ key }}">{{ value }}</option>
            {% endfor %}
        </select><br>

        <label for="age">Age (years):</label>
        <input type="number" id="age" name="age" required><br>

        <label for="mileage">Mileage (km):</label>
        <input type="number" id="mileage" name="mileage" required><br>

        <label for="engine_type">Engine Type:</label>
        <select id="engine_type" name="engine_type">
            {% for key, value in engine_types.items() %}
                <option value="{{ key }}">{{ value }}</option>
            {% endfor %}
        </select><br>

        <label for="service_level">Service Level:</label>
        <select id="service_level" name="service_level">
            {% for key, value in service_levels.items() %}
                <option value="{{ key }}">{{ value }}</option>
            {% endfor %}
        </select><br>

        <label for="parts_quality">Parts Quality:</label>
        <select id="parts_quality" name="parts_quality">
            {% for key, value in parts_qualities.items() %}
                <option value="{{ key }}">{{ value }}</option>
            {% endfor %}
        </select><br>

        <button type="submit">Predict Cost</button>
    </form>
    <div id="manual-result"></div>

    <script>
        document.getElementById('budget-form').addEventListener('submit', function(e) {
            e.preventDefault();
            fetch('/calculate', {
                method: 'POST',
                body: new FormData(this)
            }).then(response => response.json()).then(data => {
                if (data.error) {
                    document.getElementById('budget-results').innerHTML = `<p>Error: ${data.error}</p>`;
                } else {
                    let html = '<h3>Variants:</h3>';
                    data.variants.forEach(v => {
                        html += `<div><p>Car Type: ${v.car_type}, Age: ${v.age}, Mileage: ${v.mileage}, Engine: ${v.engine_type}, Service: ${v.service_level}, Parts: ${v.parts_quality}, Cost: ${v.total_cost}</p></div>`;
                    });
                    document.getElementById('budget-results').innerHTML = html;
                }
            });
        });

        document.getElementById('manual-form').addEventListener('submit', function(e) {
            e.preventDefault();
            fetch('/predict_cost', {
                method: 'POST',
                body: new FormData(this)
            }).then(response => response.json()).then(data => {
                if (data.error) {
                    document.getElementById('manual-result').innerHTML = `<p>Error: ${data.error}</p>`;
                } else {
                    document.getElementById('manual-result').innerHTML = `<p>Predicted Cost: ${data.predicted_cost}</p>`;
                }
            });
        });
    </script>
</body>
</html>
"""

# To run: Run the script, it will generate and train if needed, then run the app
if __name__ == '__main__':
    # Save the HTML template
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(HTML_TEMPLATE)

    app.run(debug=True)