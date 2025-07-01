from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
import aiosmtplib
import shutil
import os
from email.message import EmailMessage
import gdown
import traceback

# Cargar variables de entorno
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Descargar modelo si no existe
model_path = "best.pt"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1OSgQoJyItUnGlRtuW1H2na1pHZrc6qFw"
    print("📥 Descargando modelo best.pt desde Google Drive...")
    gdown.download(url, model_path, quiet=False)
else:
    print("✅ Modelo best.pt ya existe.")

# Inicializar FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLO
model = YOLO(model_path)

# Función para enviar correo
async def enviar_reporte_asincrono(asunto, cuerpo, adjunto_path=None):
    try:
        mensaje = EmailMessage()
        mensaje["From"] = EMAIL_USER
        mensaje["To"] = "appbaches@gmail.com"
        mensaje["Subject"] = asunto
        mensaje.set_content(cuerpo)

        if adjunto_path:
            with open(adjunto_path, "rb") as f:
                mensaje.add_attachment(
                    f.read(),
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(adjunto_path)
                )

        await aiosmtplib.send(
            mensaje,
            hostname="smtp.gmail.com",
            port=587,
            start_tls=True,
            username=EMAIL_USER,
            password=EMAIL_PASS
        )
        print("📧 Correo enviado con éxito.")
    except Exception as e:
        print("❌ Error al enviar correo:", str(e))
        traceback.print_exc()

# Página principal
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API de detección
@app.post("/api/detect-bache")
async def detect_bache(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        filename = f"uploads/{file.filename}"
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("📷 Imagen recibida:", filename)
        results = model.predict(source=filename, save=False, conf=0.7)
        print("🔍 Resultado YOLO:", results)

        clases = results[0].names
        detecciones = results[0].boxes.cls.tolist()
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ubicacion = "Lat: 19.4326, Lon: -99.1332"  # Ubicación simulada

        if not detecciones:
            cuerpo = f"🕓 Fecha y hora: {fecha_hora}\n📍 Ubicación: {ubicacion}\n✅ Resultado: No se detectaron baches."
            await enviar_reporte_asincrono("Reporte: Sin baches detectados", cuerpo, adjunto_path=filename)
            return {
                "tipo": "No detectado",
                "fecha_hora": fecha_hora,
                "ubicacion": ubicacion
            }

        tipos_detectados = [clases[int(i)] for i in detecciones]
        cuerpo = f"🕓 Fecha y hora: {fecha_hora}\n📍 Ubicación: {ubicacion}\n🚧 Tipos detectados: {', '.join(tipos_detectados)}"
        await enviar_reporte_asincrono("Reporte: Bache detectado", cuerpo, adjunto_path=filename)
        return {
            "tipo": ", ".join(tipos_detectados),
            "fecha_hora": fecha_hora,
            "ubicacion": ubicacion
        }
    except Exception as e:
        print("❌ Error en /api/detect-bache:", str(e))
        traceback.print_exc()
        return {"error": "Ocurrió un error en el servidor. Revisa los logs."}
