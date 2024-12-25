from flask import Flask, request, jsonify
import joblib
import pymysql
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from flask_cors import CORS  # Importar CORS
from dotenv import load_dotenv
import os
from datetime import datetime
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

# pip install flask joblib pandas scikit-learn nltk
# pip install pymysql 
# pip install python-dotenv

# Cargar variables del archivo .env
load_dotenv()
# Configurar la conexión a la base de datos
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
}
  
# Cargar el modelo del archivo .pkl en tu API o backend
componentes = joblib.load("modelo_completo.pkl")
model = componentes["modelo"]
vectorizer = componentes["vectorizador"]
label_encoder = componentes["label_encoder"]
print("Componentes cargados correctamente desde 'modelo_completo.pkl'")

# Inicializar Flask
app = Flask(__name__)

# Habilitar CORS para todas las rutas y dominios
CORS(app)

# Función para obtener una conexión a la base de datos 
def obtener_conexion():
    try:
        # Intentamos conectar utilizando PyMySQL
        conexion = pymysql.connect(**db_config)
        
        # Verificamos si la conexión fue exitosa
        print("Conexión exitosa a la base de datos")
        cursor = conexion.cursor()
        return conexion, cursor  # Devolvemos la conexión y el cursor
    except pymysql.MySQLError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None, None  # Indicamos que la conexión falló


# Función para preprocesar texto
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    
    nltk.download('stopwords')
    spanish_stopwords = stopwords.words('spanish')
    stemmer = SnowballStemmer('spanish')
    negation_words = ['no', 'nunca', 'jamás', 'ni']
    negation_flag = False
    
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü]', ' ', text)
    tokens = text.split()
    processed_tokens = []
    
    for word in tokens:
        if word in negation_words:
            negation_flag = True
            processed_tokens.append(word)
        elif negation_flag:
            processed_tokens.append('no_' + word)
            negation_flag = False
        elif word not in spanish_stopwords:
            processed_tokens.append(stemmer.stem(word))
    return ' '.join(processed_tokens)

# Función para asignar calificación en función de la clasificación
def asignar_puntajecalificacion(clasificacion):
    if clasificacion == 'Negativo':
        return 1  # Calificación de 1 | random.randint(1, 2)
    elif clasificacion == 'Neutro':
        return 3  # Calificación 3
    elif clasificacion == 'Positivo':
        return 5  # Calificación de 5 | random.randint(3, 4)
    return 0  # En caso de error

# Endpoint para clasificar comentarios
@app.route('/clasificar', methods=['POST'])
def clasificar_comentario():
 try:
        data = request.json
        print(data)
        
        # Validar datos requeridos
        product_id = data.get('product_id')
        user_id = data.get('user_id')
        user_comment = data.get('user_comment')

        if not product_id or not user_id or not user_comment:
            return jsonify({"error": "Todos los campos (product_id, user_id, user_comment) son requeridos"}), 400
        
        # Generar la fecha actual
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Conectar a la base de datos
        conexion, cursor = obtener_conexion()
        if not conexion:
            return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

        # Insertar comentario en la tabla `product_comment`
        try:
            # Inserta el comentario
            cursor.execute(
                "INSERT INTO product_comment (product_id, user_id, user_comment, date_comment) VALUES (%s, %s, %s, %s)",
                (product_id, user_id, user_comment, current_date)
            )
            conexion.commit()  # Confirma la transacción
            # Obtiene el último ID insertado
            new_comment_id = cursor.lastrowid
            print(f"Comentario registrado con éxito. ID: {new_comment_id}")
        except Exception as e:
            conexion.rollback()
            return jsonify({"error": f"Error al insertar comentario: {e}"}), 500

        # Clasificar comentario
        comentario_prep = preprocess_text(user_comment)
        comentario_tfidf = vectorizer.transform([comentario_prep])
        prediccion = model.predict(comentario_tfidf)
        clasificacion = label_encoder.inverse_transform(prediccion)[0]
        puntaje = asignar_puntajecalificacion(clasificacion)

        # Insertar clasificación en la tabla `audit_product_comment`
        try:
            cursor.execute(
                "INSERT INTO audit_product_comment (idprodcomment, product_id, user_id, user_comment, classification, rating, date_audit) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s);",
                (new_comment_id, product_id, user_id, user_comment, clasificacion, puntaje, current_date)
            )
            conexion.commit()
            print(f"Clasificación registrada para el comentario {new_comment_id}")
        except Exception as e:
            conexion.rollback()
            return jsonify({"error": f"Error al registrar clasificación: {e}"}), 500
        finally:
            conexion.close()

        # Responder con el resultado
        return jsonify({
            "message": "Comentario registrado y clasificado con éxito",
            "comment_id": new_comment_id,
            "classification": clasificacion,
            "rating": puntaje
        }), 201
 except Exception as e:
        return jsonify({"error": str(e)}), 500

# Iniciar servidor
if __name__ == '__main__':
     app.run(port=5000)
