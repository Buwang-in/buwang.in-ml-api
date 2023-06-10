from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


model = get_yolov5()

app = FastAPI(
    title="Buwang.in Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    print("input_image =>", input_image)
    results = model(input_image)
    print("results =>", results)
    a = results.render()  # updates results.imgs with boxes and labels
    print("a =>", a)
    
    object_counts = {}
    
    for img in a:
        print('img', img)
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        
        # Menghitung jumlah objek dan nama objek
        labels = results.pandas().xyxy[0]['name']
        for label in labels:
            if label not in object_counts:
                object_counts[label] = 1
            else:
                object_counts[label] += 1
    
    # Mengubah hasil perhitungan menjadi string
    object_counts_str = ", ".join([f"{label}: {count}" for label, count in object_counts.items()])
    
    # Mengembalikan gambar dan hasil perhitungan dalam format JSON
    return {
        "image": bytes_io.getvalue(),
        "object_counts": object_counts_str
    }