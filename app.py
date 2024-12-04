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
        # Obtener datos JSON del cliente
        data = request.json
        print(data)
        comentarios = data.get("comentarios", [])
        
        # Validar entrada
        if not comentarios or not isinstance(comentarios, list):
            return jsonify({"error": "Debe enviar una lista de comentarios."}), 400
        
          # Preprocesar comentarios
        resultados = []
        for item in comentarios:
            comentario = item.get("user_comment")
            if comentario:
                # Preprocesar el comentario
                comentario_prep = preprocess_text(comentario)
                
                # Vectorizar y predecir
                comentario_tfidf = vectorizer.transform([comentario_prep])
                prediccion = model.predict(comentario_tfidf)
                clasificacion = label_encoder.inverse_transform(prediccion)[0]

                # Asignar la calificación en función de la clasificación (1 al 5)
                puntaje = asignar_puntajecalificacion(clasificacion)
                
                # Recoger más información
                idprodcomment= item.get("idprodcomment", "Id del comentario del producto")
                product_id= item.get("product_id", "Id del producto")
                user_id= item.get("user_id","Id del usuario")
                date_comment=item.get("date_comment","Fecha comentario")
               
                # Formar el resultado con la predicción incluida
                resultados.append({
                    "idprodcomment":idprodcomment,
                    "product_id": product_id,
                    "user_id": user_id,
                    "user_comment": comentario,
                    "date_comment":date_comment,
                    "classification": clasificacion, # Aquí va la predicción
                    "rating": puntaje  # Aquí va el puntaje
                })
                print(resultados)

                # Conectar a la base de datos
                conexion, cursor = obtener_conexion()
                if conexion:
                    try:
                        # Insertar los datos en la base de datos
                        cursor.execute(
                            "INSERT INTO audit_product_comment (idprodcomment,product_id, user_id, user_comment, classification, rating, date_audit) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (idprodcomment,product_id, user_id, comentario, clasificacion, puntaje, date_comment)
                        )
                        conexion.commit()  # Confirmar cambios
                        print(f"idprodcomment: {idprodcomment}, Producto ID: {product_id}, Usuario ID: {user_id}, Comentario: {comentario}, Clasificación: {clasificacion}, Fecha: {date_comment}")
                    except Exception as e:
                        print(f"Error al insertar en la base de datos: {e}")
                    finally:
                        # Cerrar la conexión
                        conexion.close()

          # Devolver los resultados
        return jsonify(resultados), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Iniciar servidor
if __name__ == '__main__':
     app.run(port=5000)
