import onnx


model = onnx.load("./model/model.onnx")
print(onnx.helper.printable_graph(model.graph))
