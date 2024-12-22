# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', default='fallback-key-if-none-found')

# Mock User class (you may use Flask-Login if preferred)
class User:
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

class ODGJPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.poly = None
        
    def load_data(self):
        # Load dataset and aggregate yearly data
        df = pd.read_csv(self.csv_path)
        yearly_data = df.groupby('tahun')['total_gangguan_jiwa'].sum().reset_index()
        return yearly_data
        
    def train_model(self):
        data = self.load_data()
        X = data['tahun'].values.reshape(-1, 1)
        y = data['total_gangguan_jiwa'].values
        print(f"Jumlah data: {len(X)}")

        if len(X) < 2:
            raise ValueError("Dataset terlalu kecil untuk melakukan train-test split. Tambahkan lebih banyak data.")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Polynomial features
        self.poly = PolynomialFeatures(degree=1)
        X_train_poly = self.poly.fit_transform(X_train)
        X_val_poly = self.poly.transform(X_val)
        X_test_poly = self.poly.transform(X_test)

        # Train model with Ridge Regression
        self.model = Ridge(alpha=100)  # Anda dapat menyesuaikan nilai alpha
        self.model.fit(X_train_poly, y_train)

        # Evaluasi
        train_pred = self.model.predict(X_train_poly)
        val_pred = self.model.predict(X_val_poly)
        test_pred = self.model.predict(X_test_poly)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        print(f"Training MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")
        print(f"Test MSE: {test_mse}")

            # Hitung MAE relatif
        mae = mean_absolute_error(y_test, test_pred)
        average_prediction = np.mean(test_pred)
        mae_relative = (mae / average_prediction) * 100

        # Cetak hasil
        print(f"MAE: {mae}")
        print(f"Rata-rata prediksi: {average_prediction}")
        print(f"MAE Relatif: {mae_relative:.2f}%")
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(self.model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_mse = -np.mean(scores)
        print(f"Cross-Validation MSE: {mean_mse}")

        # Calculate metrics
        metrics = {
            'train': self.calculate_metrics(X_train, y_train, is_training=True),
            'validation': self.calculate_metrics(X_val, y_val),
            'test': self.calculate_metrics(X_test, y_test)
        }

        return metrics, data


    def calculate_metrics(self, X, y, is_training=False):
        if is_training:
            X_poly = self.poly.fit_transform(X)  # Hanya fit pada data training
        else:
            X_poly = self.poly.transform(X)
        y_pred = self.model.predict(X_poly)
        
        return {
            'mse': float(mean_squared_error(y, y_pred)),
            'mae': float(mean_absolute_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred)))
        }
        
    def predict_future(self, years):
        future_years = np.array(years).reshape(-1, 1)
        future_years_poly = self.poly.transform(future_years)
        predictions = self.model.predict(future_years_poly)
        
        return predictions.tolist()

# Path to the combined dataset
predictor = ODGJPredictor('dataset_vertikal.csv')

# @login_manager.user_loader
# def load_user(user_id):
#     response = supabase.table("register").select("*").eq("id", user_id).execute()
#     if len(response.data) == 0:
#         return None
#     user_data = response.data[0]
#     return User(id=user_data['id'], username=user_data['username'], password=user_data['password'])


@app.route('/plot.png')
def plot():
    import matplotlib
    matplotlib.use('Agg')  # Ganti backend ke non-GUI untuk server
    import matplotlib.pyplot as plt
    import io

    metrics, historical_data = predictor.train_model()
    future_years = list(range(2014, 2031))
    predictions = predictor.predict_future(future_years)

    # Prepare data for plotting
    historical_years = historical_data['tahun']
    historical_totals = historical_data['total_gangguan_jiwa']

    # Plot data historis
    plt.figure(figsize=(10, 6))
    plt.scatter(historical_years, historical_totals, color='blue', label='Data Historis')
    
    # Plot prediksi
    plt.plot(future_years, predictions, color='red', label='Prediksi', linestyle='--')

    # Add labels and title
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Gangguan Jiwa')
    plt.title('Prediksi Jumlah Gangguan Jiwa')
    plt.legend()
    plt.grid(True)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/')
def home():
    user = session.get('user')
    return render_template('home.html', user=user)

@app.route('/about')
def about():
    return render_template('about.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password using pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Check if the user already exists
        response = supabase.table("register").select("*").eq("username", username).execute()
        if response.data:
            flash("Username already exists", "danger")
            return redirect(url_for('register'))

        # Insert user into the database
        supabase.table("register").insert({
            "name": name,
            "username": username,
            "email": email,
            "password": hashed_password
        }).execute()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verify user credentials
        response = supabase.table("register").select("*").eq("username", username).execute()
        if not response.data:
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))
        
        user = response.data[0]
        if not check_password_hash(user['password'], password):
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))

        # Store user information in session
        session['logged_in'] = True
        session['user'] = {
            "id": user['id'],
            "username": user['username'],
            "email": user['email']
        }

        flash("Login successful!", "success")
        return redirect(url_for('dashboard'))

    return render_template('login.html')

# Dashboard route (protected)
@app.route('/dashboard')
def dashboard():
    # Check if user is logged in by checking the session
    if not session.get('logged_in'):
        flash("You need to log in to access this page.", "danger")
        return redirect(url_for('login'))

    # Access user data from the session
    user = session.get('user')

    # Call your predictor logic
    metrics, historical_data = predictor.train_model()
    future_years = list(range(2024, 2031))
    predictions = predictor.predict_future(future_years)

    # Prepare data for template
    historical_data_list = historical_data.to_dict('records')
    prediction_data = [{'tahun': year, 'total_gangguan_jiwa': int(value)}
                       for year, value in zip(future_years, predictions)]

    return render_template('index.html',
                           historical_data=historical_data_list,
                           prediction_data=prediction_data,
                           metrics=metrics,
                           user=user)


# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
