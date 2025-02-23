# app.py
from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def crear_app():
    app = Flask(__name__, static_folder='templates')

    @app.route('/')
    def home():
        return app.send_static_file('index.html')

    @app.route('/analyze', methods=['POST'])
    def analyze():
        try:
            data = request.json
            x_values = np.array([float(x) for x in data['x']]).reshape(-1, 1)
            y_values = np.array([float(y) for y in data['y']])

            # Realizar regresión lineal
            model = LinearRegression()
            model.fit(x_values, y_values)

            # Calcular R² y RMSE
            y_pred = model.predict(x_values)
            r_squared = r2_score(y_values, y_pred)
            mse = mean_squared_error(y_values, y_pred)
            rmse = np.sqrt(mse)

            # Calcular puntos para la línea de regresión
            x_line = np.array([min(x_values), max(x_values)]).reshape(-1, 1)
            y_line = model.predict(x_line)
            y_line_plus_rmse = y_line + rmse
            y_line_minus_rmse = y_line - rmse

            # Calcular estadísticas
            def calculate_stats(values):
                return {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': np.ptp(values),
                    'q1': np.percentile(values, 25),
                    'median': np.median(values),
                    'q3': np.percentile(values, 75)
                }

            stats = {
                'x': calculate_stats(x_values.flatten()),
                'y': calculate_stats(y_values)
            }

            return jsonify({
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(r_squared),
                'mse': float(mse),
                'rmse': float(rmse),
                'regression_line': {
                    'x': x_line.flatten().tolist(),
                    'y': y_line.tolist(),
                    'y_plus_rmse': y_line_plus_rmse.tolist(),
                    'y_minus_rmse': y_line_minus_rmse.tolist()
                },
                'stats': stats,
                'x': x_values.flatten().tolist(),
                'y': y_values.tolist()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app

if __name__ == '__main__':
    app = crear_app()
    app.run(debug=True)
