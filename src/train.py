from ultralytics import YOLO
import uuid

training_run = str(uuid.uuid1())

#Initialize model
model = YOLO("model/yolo26n-seg.pt")

print(model)
results = model.train(
    data = 'src/yolo26-seg.yaml', #YAML train
    epochs = 100,
    patience = 10,
    batch = -1,
    imgsz = 640,
    device ="mps",
    cache = True,
    workers =6,
    project='experiments',
    name = training_run,
    seed = 42,
    verbose = True,
    freeze = 9
)

model.save(f'model_{training_run}.pt')
model.export(format="onnx")

