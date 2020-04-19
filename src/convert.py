import cntk as C

model_path = "./out/model_98.cntk"
z = C.Function.load(model_path, device=C.device.cpu())

z.save("model.onnx", format=C.ModelFormat.ONNX)
