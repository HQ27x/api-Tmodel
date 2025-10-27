# API de Reconocimiento de Tumores Cerebrales 🧠

API REST construida con FastAPI para clasificar imágenes de resonancias magnéticas cerebrales utilizando un modelo EfficientNet-B3 entrenado en 44 clases de tumores.

## 🚀 Características

- **Predicción Individual**: Sube una imagen y obtén la clasificación
- **Predicción por Lotes**: Procesa hasta 10 imágenes simultáneamente
- **Top 3 Predicciones**: Obtén las 3 clases más probables con sus confianzas
- **Documentación Interactiva**: Swagger UI integrado
- **CORS Habilitado**: Listo para consumir desde aplicaciones web
- **Health Check**: Endpoint para monitoreo

## 📋 Requisitos Previos

- Python 3.11+
- Modelo entrenado (`.pth` file)

## 🛠️ Instalación Local

1. **Clonar el repositorio**
```bash
git clone <tu-repositorio>
cd modeloapi
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **IMPORTANTE: Actualizar los nombres de las clases**

Edita el archivo `app.py` y reemplaza la lista `CLASS_NAMES` con los nombres reales de tus 44 clases de tumores en el orden correcto.

5. **Ejecutar la API**
```bash
python app.py
```

La API estará disponible en `http://localhost:8000`

## 📚 Documentación de la API

Una vez que la API esté corriendo, visita:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔌 Endpoints

### 1. Verificar Estado
```bash
GET /
```
Respuesta:
```json
{
  "mensaje": "API de Reconocimiento de Tumores Cerebrales",
  "estado": "activa",
  "version": "1.0.0"
}
```

### 2. Health Check
```bash
GET /health
```

### 3. Obtener Clases
```bash
GET /classes
```
Retorna todas las clases que el modelo puede predecir.

### 4. Predicción Individual
```bash
POST /predict
Content-Type: multipart/form-data
```

**Parámetros:**
- `file`: Archivo de imagen (JPEG, PNG, etc.)

**Ejemplo con cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@imagen_tumor.jpg"
```

**Respuesta:**
```json
{
  "exito": true,
  "clase_predicha": "Germinoma T1C+",
  "confianza": 95.67,
  "top_3_predicciones": [
    {
      "clase": "Germinoma T1C+",
      "confianza": 95.67
    },
    {
      "clase": "Astrocytoma T1",
      "confianza": 3.12
    },
    {
      "clase": "Meningioma T2",
      "confianza": 0.89
    }
  ]
}
```

### 5. Predicción por Lotes
```bash
POST /predict/batch
Content-Type: multipart/form-data
```

**Parámetros:**
- `files`: Lista de archivos de imagen (máximo 10)

## ☁️ Despliegue en Render

### Opción 1: Despliegue desde GitHub (Recomendado)

1. **Subir el código a GitHub**
```bash
git init
git add .
git commit -m "Initial commit: Brain tumor detection API"
git branch -M main
git remote add origin <tu-repositorio-github>
git push -u origin main
```

2. **Crear cuenta en Render**
- Ve a [render.com](https://render.com) y crea una cuenta

3. **Crear nuevo Web Service**
- Click en "New +" → "Web Service"
- Conecta tu repositorio de GitHub
- Configura el servicio:

**Configuración:**
- **Name**: `brain-tumor-api` (o el nombre que prefieras)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free (o el plan que prefieras)

4. **Variables de Entorno (opcional)**
En "Environment Variables" puedes agregar:
- `PYTHON_VERSION`: `3.11.6`

5. **Deploy**
- Click en "Create Web Service"
- Render automáticamente construirá y desplegará tu API

### Opción 2: Despliegue Manual

Si prefieres no usar GitHub, puedes usar Render CLI o Docker.

## 📱 Consumir la API desde una App

### Ejemplo JavaScript (React/Next.js)

```javascript
async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('https://tu-api.onrender.com/predict', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('Predicción:', result.clase_predicha);
  console.log('Confianza:', result.confianza + '%');
  return result;
}
```

### Ejemplo Python

```python
import requests

url = "https://tu-api.onrender.com/predict"
files = {'file': open('imagen_tumor.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Clase: {result['clase_predicha']}")
print(f"Confianza: {result['confianza']}%")
```

### Ejemplo cURL

```bash
curl -X POST "https://tu-api.onrender.com/predict" \
  -F "file=@imagen_tumor.jpg"
```

## ⚠️ Consideraciones Importantes

1. **Nombres de Clases**: Debes actualizar la lista `CLASS_NAMES` en `app.py` con los nombres exactos de tus 44 clases en el orden correcto.

2. **Tamaño del Modelo**: El modelo ocupa ~50-100MB. Render permite hasta 512MB en el plan gratuito, así que debería funcionar bien.

3. **Cold Start**: En el plan gratuito de Render, la API puede "dormirse" después de 15 minutos de inactividad. La primera petición después puede tardar 30-60 segundos.

4. **Limitaciones del Plan Gratuito**:
   - 512 MB RAM
   - 750 horas de servicio por mes
   - La instancia se duerme después de 15 min sin uso
   - Considera un plan pago para producción

5. **Seguridad**: Por defecto, CORS está abierto (`allow_origins=["*"]`). En producción, especifica solo los dominios permitidos.

## 🧪 Testing

Para probar localmente:

```bash
# Iniciar servidor
python app.py

# En otra terminal, probar con una imagen
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

## 📊 Monitoreo

Render proporciona logs en tiempo real:
- Ve a tu servicio en Render Dashboard
- Click en "Logs" para ver los logs de la aplicación

## 🐛 Solución de Problemas

### Error: "Modelo no cargado"
- Verifica que el archivo `.pth` esté en la carpeta `modelo/`
- Asegúrate de que el nombre del archivo coincida con el especificado en `app.py`

### Error: "Out of memory"
- El modelo es demasiado grande para el plan gratuito
- Considera usar un plan con más RAM o optimizar el modelo

### La API es lenta
- En el plan gratuito, la primera predicción después de "despertar" será lenta
- Considera mantener la API "despierta" con pings regulares

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

## 👥 Autores

- Tu nombre aquí

## 🙏 Agradecimientos

- Modelo basado en EfficientNet-B3
- Framework: FastAPI
- Hosting: Render

---

¿Necesitas ayuda? Abre un issue en el repositorio.

