# buwang.in-ml-api

This project is for creating the API for the machine learning model using FAST API and YOLOv5


### Getting start for this project

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Documentation

This project is have 3 endpoint for the API :
1. /notify/v1/health

    This endpoint is for checking the health of the API. The response will be like this :
    ```
    {
        "status": "OK"
    }
    ```

2. /object-to-json

    This endpoint is for getting the object detection result from the image. The response will be like this :
    ```
    {
        "result": [
            {
                "class": "person",
                "confidence": 0.999,
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 0.0,
                "ymax": 0.0
            }
        ]
    }
    ```
3. /object-to-img

    This endpoint is for getting the image with the bounding box from the image and count labels object

