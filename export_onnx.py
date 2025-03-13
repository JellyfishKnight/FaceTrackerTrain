import torch
import torch.nn as nn
import argparse
import onnx
import onnxruntime
from model import FacialActionModelWithAuxLoss  # 确保导入你的模型类

def load_model(pth_path, num_classes=43, device="cpu"):
    """加载 PyTorch 模型"""
    model = FacialActionModelWithAuxLoss(num_classes=num_classes, backbone='shufflenet', pretrained=True)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()  # 设为推理模式
    print(f"✅ 模型 {pth_path} 加载完成")
    return model

def export_to_onnx(model, onnx_path, input_size=(1, 3, 224, 224), device="cpu"):
    """导出 PyTorch 模型为 ONNX"""
    dummy_input = torch.randn(*input_size).to(device)  # 创建随机输入
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,  # 保存参数
        opset_version=11,  # ONNX 版本
        do_constant_folding=True,  # 常量折叠优化
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 允许动态 batch
    )
    print(f"🎉 成功导出 ONNX: {onnx_path}")

def verify_onnx(onnx_path):
    """验证 ONNX 模型"""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"✅ ONNX 模型 {onnx_path} 通过验证")

def run_onnx_inference(onnx_path, input_size=(1, 3, 224, 224)):
    """使用 ONNX Runtime 运行推理测试"""
    ort_session = onnxruntime.InferenceSession(onnx_path)
    dummy_input = torch.randn(*input_size).numpy()
    
    outputs = ort_session.run(None, {"input": dummy_input})
    print(f"🔍 ONNX 推理成功，输出形状: {outputs[0].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, required=True, help="输入的 PyTorch .pth 文件路径")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="输出 ONNX 文件路径")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="运行设备")
    args = parser.parse_args()

    # 加载 PyTorch 模型
    model = load_model(args.pth, device=args.device)

    # 导出 ONNX
    export_to_onnx(model, args.onnx, device=args.device)

    # 验证 ONNX
    verify_onnx(args.onnx)

    # 运行 ONNX 推理测试
    run_onnx_inference(args.onnx)
