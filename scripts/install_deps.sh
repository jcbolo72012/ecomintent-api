#!/bin/bash
set -e
echo "Installing EcomIntent dependencies..."

pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0
pip install datasets==3.2.0
pip install peft==0.14.0
pip install trl==0.13.0
pip install accelerate==1.2.0
pip install evaluate==0.4.3
pip install scikit-learn==1.6.0
pip install seaborn==0.13.2
pip install matplotlib==3.10.0

# Data processing
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install pyarrow==18.1.0

# API serving
pip install fastapi==0.115.6
pip install uvicorn==0.34.0
pip install python-multipart==0.0.20
pip install pydantic==2.10.4
pip install slowapi==0.1.9

# Deployment
pip install modal==0.73.0
pip install huggingface_hub==0.27.0

# Utilities
pip install anthropic==0.42.0
pip install python-dotenv==1.0.1
pip install rich==13.9.4
pip install tqdm==4.67.1

echo "All dependencies installed."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
