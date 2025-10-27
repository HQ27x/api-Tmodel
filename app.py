"""
API para el modelo de reconocimiento de tumores cerebrales
Basado en EfficientNet-B3 entrenado en 44 clases
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la arquitectura del modelo (debe ser idéntica al entrenamiento)
class BrainTumorEfficientNet(nn.Module):
    def __init__(self, num_classes=44):
        super(BrainTumorEfficientNet, self).__init__()
        
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x


# Nombres de las clases (las 44 clases del modelo)
# Clases reales del dataset de tumores cerebrales
CLASS_NAMES = [
    "Astrocitoma T1",
    "Astrocitoma T1C+",
    "Astrocitoma T2",
    "Carcinoma T1",
    "Carcinoma T1C+",
    "Carcinoma T2",
    "Ependimoma T1",
    "Ependimoma T1C+",
    "Ependimoma T2",
    "Ganglioglioma T1",
    "Ganglioglioma T1C+",
    "Ganglioglioma T2",
    "Germinoma T1",
    "Germinoma T1C+",
    "Germinoma T2",
    "Glioblastoma T1",
    "Glioblastoma T1C+",
    "Glioblastoma T2",
    "Granuloma T1",
    "Granuloma T1C+",
    "Granuloma T2",
    "Meduloblastoma T1",
    "Meduloblastoma T1C+",
    "Meduloblastoma T2",
    "Meningioma T1",
    "Meningioma T1C+",
    "Meningioma T2",
    "Neurocitoma T1",
    "Neurocitoma T1C+",
    "Neurocitoma T2",
    "Oligodendroglioma T1",
    "Oligodendroglioma T1C+",
    "Oligodendroglioma T2",
    "Papiloma T1",
    "Papiloma T1C+",
    "Papiloma T2",
    "Schwannoma T1",
    "Schwannoma T1C+",
    "Schwannoma T2",
    "Tuberculoma T1",
    "Tuberculoma T1C+",
    "Tuberculoma T2",
    "_NORMAL T1",
    "_NORMAL T2"
]

# Inicializar FastAPI
app = FastAPI(
    title="API de Reconocimiento de Tumores Cerebrales",
    description="API para clasificar imágenes de resonancias magnéticas cerebrales en 44 categorías",
    version="1.0.0"
)

# Configurar CORS para permitir llamadas desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
device = None
model = None

# Transformaciones para las imágenes (deben coincidir con el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


@app.on_event("startup")
async def load_model():
    """Cargar el modelo al iniciar la aplicación"""
    global device, model
    
    try:
        # Detectar si hay GPU disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        
        # Cargar el modelo
        model = BrainTumorEfficientNet(num_classes=44).to(device)
        
        # Cargar los pesos del modelo entrenado
        # IMPORTANTE: Ajusta la ruta según donde esté tu modelo
        model_path = "modelo/brain_tumor_model.pth_epoch20.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info("✅ Modelo cargado exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {str(e)}")
        raise


@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "mensaje": "API de Reconocimiento de Tumores Cerebrales",
        "estado": "activa",
        "version": "1.0.0",
        "documentacion": "/docs"
    }


@app.get("/health")
async def health_check():
    """Verificar el estado de salud de la API"""
    return {
        "estado": "saludable",
        "modelo_cargado": model is not None,
        "dispositivo": str(device)
    }


@app.get("/classes")
async def get_classes():
    """Obtener la lista de todas las clases que el modelo puede predecir"""
    return {
        "total_clases": len(CLASS_NAMES),
        "clases": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint principal para hacer predicciones
    
    Parámetros:
    - file: Archivo de imagen (JPEG, PNG, etc.)
    
    Retorna:
    - clase_predicha: Nombre de la clase con mayor probabilidad
    - confianza: Probabilidad de la predicción (0-100)
    - top_3_predicciones: Las 3 clases más probables con sus probabilidades
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Leer la imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Aplicar transformaciones
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Hacer la predicción
        with torch.no_grad():
            output = model(image_tensor)
            # Convertir log-probabilities a probabilidades
            probabilities = torch.exp(output)
            
            # Obtener las 3 predicciones más probables
            top3_prob, top3_classes = torch.topk(probabilities, 3, dim=1)
            
            # Predicción principal
            predicted_class_idx = top3_classes[0][0].item()
            confidence = top3_prob[0][0].item() * 100
        
        # Preparar respuesta con top 3
        top3_predictions = []
        for i in range(3):
            class_idx = top3_classes[0][i].item()
            prob = top3_prob[0][i].item() * 100
            top3_predictions.append({
                "clase": CLASS_NAMES[class_idx],
                "confianza": round(prob, 2)
            })
        
        logger.info(f"Predicción exitosa: {CLASS_NAMES[predicted_class_idx]} ({confidence:.2f}%)")
        
        return JSONResponse({
            "exito": True,
            "clase_predicha": CLASS_NAMES[predicted_class_idx],
            "confianza": round(confidence, 2),
            "top_3_predicciones": top3_predictions
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint para hacer predicciones en múltiples imágenes
    
    Parámetros:
    - files: Lista de archivos de imagen
    
    Retorna:
    - Lista de predicciones para cada imagen
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Máximo 10 imágenes por solicitud")
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            # Leer la imagen
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Aplicar transformaciones
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Hacer la predicción
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.exp(output)
                predicted_class_idx = probabilities.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item() * 100
            
            results.append({
                "nombre_archivo": file.filename,
                "clase_predicha": CLASS_NAMES[predicted_class_idx],
                "confianza": round(confidence, 2)
            })
            
        except Exception as e:
            results.append({
                "nombre_archivo": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "exito": True,
        "total_imagenes": len(files),
        "resultados": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

