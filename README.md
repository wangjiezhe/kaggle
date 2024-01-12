Code for Kaggle competitions
============================

1. [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. [Classify Leaves](https://www.kaggle.com/c/classify-leaves)
3. [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
4. [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)


Annotation for transforming code to run on Kaggle
-------------------------------------------------

### GPU selection

- Train with full precision: choose `P100`.
- Train with half precision: choose `T4 x2`.

### Use multiple GPU

- Add `strategy="ddp_notebook"` to `pl.Trainer`.
- Use another trainer with `devices=1` for test and predict.
- Add `sync_dist=True` to `log` and `log_dict`.
- `trainer.fit` can only called once.
- Fix warnings with `permute` and `transpose` when using DDP, issued in [#47163](https://github.com/pytorch/pytorch/issues/47163), as shown below:

```bash
sed -i 's#\(permute(.*\?).*\)#\1.contiguous()#' \
    /opt/conda/lib/python3.10/site-packages/torchvision/models/convnext.py \
    /opt/conda/lib/python3.10/site-packages/torchvision/ops/misc.py \
    ...
```

### Torchvision

- The version of `torchvision` is 0.15.1, so there are not `v2.CutMix` and `v2.MixUp` in `torchvision.transforms`.
- Use `import torchvision.transforms as v1` instead.

### Use Tensorboard

```bash
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && \
    sudo apt update && sudo apt install ngrok
```

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
ngrok_authtoken = user_secrets.get_secret("NGROK_AUTHTOKEN")
```

```python
from subprocess import Popen
import time

Popen(f"ngrok config add-authtoken {ngrok_authtoken}", shell=True)
time.sleep(1)
```

```python
Popen("ngrok http 6006", shell=True)
time.sleep(1)
```

```bash
curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print('Tensorboard URL:', json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

```python
Popen("tensorboard --logdir ./lightning_logs/ --host 0.0.0.0 --port 6006", shell=True)
```

```bash
ps aux | grep tensorboard
```