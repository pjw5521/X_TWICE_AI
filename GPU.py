# -*- coding: utf-8 -*-
"""
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
"""

import torch

print(torch.cuda.is_available()) # cuda 사용 가능?
print(torch.cuda.device_count()) # 갯수
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
