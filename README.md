Code for Kaggle competitions
============================

1. [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. [Classify Leaves](https://www.kaggle.com/c/classify-leaves)
3. [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
4. [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)
5. [CowBoy Outfits Detection](https://www.kaggle.com/c/cowboyoutfits)


Annotation for transforming code to run on Kaggle
-------------------------------------------------

### GPU selection

- Train with full precision: choose `P100`.
- Train with half precision: choose `T4 x2`.

### Use multiple GPU

- Add `accelerator="gpu", devices=2, strategy="ddp_notebook"` to `pl.Trainer`.
- Use another trainer with `devices=1` for test and predict.
- Add `sync_dist=True` to `log` and `log_dict`.
- `trainer.fit` can only called once.
- When rewriting any epoch_end function, if you log, just make sure that the tensor is on gpu device. If you initialize new tensor, initialize it with device=self.device. See [#18803](https://github.com/Lightning-AI/pytorch-lightning/issues/18803).
- Fix warnings with `permute` and `transpose` when using DDP, issued in [#47163](https://github.com/pytorch/pytorch/issues/47163), as shown below:

```bash
sed -i 's#\(permute(.*\?)\)#\1.contiguous()#' \
    /opt/conda/lib/python3.10/site-packages/torchvision/models/convnext.py \
    /opt/conda/lib/python3.10/site-packages/torchvision/ops/misc.py \
    /opt/conda/lib/python3.10/site-packages/torchvision/models/detection/roi_heads.py \
    /opt/conda/lib/python3.10/site-packages/torchvision/models/detection/rpn.py \
    ...
```

~~### Torchvision~~

~~- The version of `torchvision` is 0.15.1, so there are not `v2.CutMix` and `v2.MixUp` in `torchvision.transforms`.~~
~~- Use `import torchvision.transforms as v1` instead.~~

### Use Tensorboard

```bash
pip install pyngrok
```

```python
## Attach NGROK_AUTHTOKEN in `Add-ons > Secrets` first
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
ngrokToken = user_secrets.get_secret("NGROK_AUTHTOKEN")
```

```python
from pyngrok import conf, ngrok
conf.get_default().auth_token = ngrokToken
conf.get_default().monitor_thread = False
ssh_tunnels = ngrok.get_tunnels(conf.get_default())
if len(ssh_tunnels) == 0:
    ssh_tunnel = ngrok.connect(6006)
    print('address：'+ssh_tunnel.public_url)
else:
    print('address：'+ssh_tunnels[0].public_url)
```

```python
from subprocess import Popen

Popen("tensorboard --logdir ./lightning_logs/ --host 0.0.0.0 --port 6006", shell=True)
```

```bash
ps aux | grep tensorboard
```