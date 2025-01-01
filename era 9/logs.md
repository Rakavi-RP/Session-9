Using device: cuda
/tmp/ipykernel_10287/850731890.py:35: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(
Using 4 GPUs!

Epoch 1/250
Epoch 1/250 Training | loss: 0.044 | acc: 5.33% | lr: 4.04e-03  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [27:12<00:00,  6.13it/s]
Validating: 100%|████████████████████| 391/391 [00:55<00:00,  7.07it/s, loss=0.044, accuracy=12.49%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 5.33%, Val acc: 12.49%

Epoch 2/250
Epoch 2/250 Training | loss: 0.033 | acc: 17.70% | lr: 4.17e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:48<00:00,  6.22it/s]
Validating: 100%|████████████████████| 391/391 [00:34<00:00, 11.27it/s, loss=0.034, accuracy=27.06%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 17.70%, Val acc: 27.06%

Epoch 3/250
Epoch 3/250 Training | loss: 0.028 | acc: 27.28% | lr: 4.38e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:59<00:00,  6.18it/s]
Validating: 100%|████████████████████| 391/391 [00:35<00:00, 10.92it/s, loss=0.024, accuracy=35.66%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 27.28%, Val acc: 35.66%

Epoch 4/250
Epoch 4/250 Training | loss: 0.025 | acc: 33.67% | lr: 4.67e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:57<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.82it/s, loss=0.021, accuracy=41.08%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 33.67%, Val acc: 41.08%

Epoch 5/250
Epoch 5/250 Training | loss: 0.023 | acc: 38.09% | lr: 5.05e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:53<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.77it/s, loss=0.019, accuracy=45.02%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 38.09%, Val acc: 45.02%

Epoch 6/250
Epoch 6/250 Training | loss: 0.021 | acc: 41.34% | lr: 5.51e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.74it/s, loss=0.018, accuracy=48.40%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 41.34%, Val acc: 48.40%

Epoch 7/250
Epoch 7/250 Training | loss: 0.020 | acc: 43.76% | lr: 6.05e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:56<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.83it/s, loss=0.017, accuracy=50.59%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 43.76%, Val acc: 50.59%

Epoch 8/250
Epoch 8/250 Training | loss: 0.019 | acc: 45.59% | lr: 6.67e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:53<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.84it/s, loss=0.016, accuracy=52.63%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 45.59%, Val acc: 52.63%

Epoch 9/250
Epoch 9/250 Training | loss: 0.019 | acc: 46.95% | lr: 7.37e-03  : 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:53<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.82it/s, loss=0.016, accuracy=52.86%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 46.95%, Val acc: 52.86%

Epoch 10/250
Epoch 10/250 Training | loss: 0.018 | acc: 48.01% | lr: 8.15e-03  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.39it/s, loss=0.015, accuracy=54.53%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 48.01%, Val acc: 54.53%

Epoch 11/250
Epoch 11/250 Training | loss: 0.018 | acc: 48.70% | lr: 9.01e-03  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:53<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.73it/s, loss=0.015, accuracy=54.47%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 48.70%, Val acc: 54.47%

Epoch 12/250
Epoch 12/250 Training | loss: 0.018 | acc: 49.33% | lr: 9.94e-03  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:39<00:00,  9.84it/s, loss=0.015, accuracy=54.91%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 49.33%, Val acc: 54.91%

Epoch 13/250
Epoch 13/250 Training | loss: 0.018 | acc: 49.74% | lr: 1.09e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.46it/s, loss=0.014, accuracy=56.33%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 49.74%, Val acc: 56.33%

Epoch 14/250
Epoch 14/250 Training | loss: 0.017 | acc: 50.06% | lr: 1.20e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:39<00:00,  9.93it/s, loss=0.014, accuracy=56.56%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 50.06%, Val acc: 56.56%

Epoch 15/250
Epoch 15/250 Training | loss: 0.017 | acc: 50.32% | lr: 1.32e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.71it/s, loss=0.014, accuracy=56.17%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.32%, Val acc: 56.17%

Epoch 16/250
Epoch 16/250 Training | loss: 0.017 | acc: 50.57% | lr: 1.44e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:49<00:00,  6.22it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.73it/s, loss=0.015, accuracy=55.64%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.57%, Val acc: 55.64%

Epoch 17/250
Epoch 17/250 Training | loss: 0.017 | acc: 50.77% | lr: 1.57e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.52it/s, loss=0.015, accuracy=55.66%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.77%, Val acc: 55.66%

Epoch 18/250
Epoch 18/250 Training | loss: 0.017 | acc: 50.81% | lr: 1.70e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:58<00:00,  6.18it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.69it/s, loss=0.015, accuracy=55.99%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.81%, Val acc: 55.99%

Epoch 19/250
Epoch 19/250 Training | loss: 0.017 | acc: 50.87% | lr: 1.84e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:56<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.67it/s, loss=0.015, accuracy=56.39%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.87%, Val acc: 56.39%

Epoch 20/250
Epoch 20/250 Training | loss: 0.017 | acc: 50.97% | lr: 1.99e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:53<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.58it/s, loss=0.015, accuracy=55.53%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.97%, Val acc: 55.53%

Epoch 21/250
Epoch 21/250 Training | loss: 0.017 | acc: 50.87% | lr: 2.14e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:52<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.75it/s, loss=0.014, accuracy=57.06%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 50.87%, Val acc: 57.06%

Epoch 22/250
Epoch 22/250 Training | loss: 0.017 | acc: 50.96% | lr: 2.30e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:52<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.57it/s, loss=0.014, accuracy=56.42%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.96%, Val acc: 56.42%

Epoch 23/250
Epoch 23/250 Training | loss: 0.017 | acc: 50.91% | lr: 2.46e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:51<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.71it/s, loss=0.014, accuracy=56.86%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.91%, Val acc: 56.86%

Epoch 24/250
Epoch 24/250 Training | loss: 0.017 | acc: 50.85% | lr: 2.63e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:56<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:38<00:00, 10.13it/s, loss=0.014, accuracy=57.30%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 50.85%, Val acc: 57.30%

Epoch 25/250
Epoch 25/250 Training | loss: 0.017 | acc: 50.79% | lr: 2.80e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.38it/s, loss=0.015, accuracy=55.43%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.79%, Val acc: 55.43%

Epoch 26/250
Epoch 26/250 Training | loss: 0.017 | acc: 50.73% | lr: 2.98e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:52<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.77it/s, loss=0.015, accuracy=55.31%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.73%, Val acc: 55.31%

Epoch 27/250
Epoch 27/250 Training | loss: 0.017 | acc: 50.70% | lr: 3.16e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.85it/s, loss=0.014, accuracy=57.18%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.70%, Val acc: 57.18%

Epoch 28/250
Epoch 28/250 Training | loss: 0.017 | acc: 50.60% | lr: 3.34e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.69it/s, loss=0.015, accuracy=56.25%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.60%, Val acc: 56.25%

Epoch 29/250
Epoch 29/250 Training | loss: 0.017 | acc: 50.53% | lr: 3.53e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.68it/s, loss=0.015, accuracy=55.52%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.53%, Val acc: 55.52%

Epoch 30/250
Epoch 30/250 Training | loss: 0.017 | acc: 50.48% | lr: 3.72e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:51<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.48it/s, loss=0.015, accuracy=56.53%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.48%, Val acc: 56.53%

Epoch 31/250
Epoch 31/250 Training | loss: 0.017 | acc: 50.34% | lr: 3.91e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:55<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.77it/s, loss=0.014, accuracy=56.18%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.34%, Val acc: 56.18%

Epoch 32/250
Epoch 32/250 Training | loss: 0.017 | acc: 50.23% | lr: 4.10e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:56<00:00,  6.19it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.59it/s, loss=0.015, accuracy=55.06%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.23%, Val acc: 55.06%

Epoch 33/250
Epoch 33/250 Training | loss: 0.017 | acc: 50.22% | lr: 4.30e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:52<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.64it/s, loss=0.014, accuracy=56.90%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.22%, Val acc: 56.90%

Epoch 34/250
Epoch 34/250 Training | loss: 0.017 | acc: 50.10% | lr: 4.50e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:39<00:00,  9.99it/s, loss=0.015, accuracy=55.57%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.10%, Val acc: 55.57%

Using device: cuda
/tmp/ipykernel_10287/417164645.py:35: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(
/tmp/ipykernel_10287/3148022116.py:17: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path)
Using 4 GPUs!

Loading checkpoint: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth
Resumed from epoch 34 with best accuracy: 57.30%

Epoch 35/250
Epoch 35/250 Training | loss: 0.017 | acc: 50.00% | lr: 4.70e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:54<00:00,  6.20it/s]
Validating: 100%|████████████████████| 391/391 [00:39<00:00,  9.95it/s, loss=0.015, accuracy=55.80%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 50.00%, Val acc: 55.80%

Epoch 36/250
Epoch 36/250 Training | loss: 0.018 | acc: 49.91% | lr: 4.90e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:50<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:37<00:00, 10.39it/s, loss=0.015, accuracy=54.27%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 49.91%, Val acc: 54.27%

Epoch 37/250
Epoch 37/250 Training | loss: 0.018 | acc: 49.91% | lr: 5.10e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:50<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:36<00:00, 10.76it/s, loss=0.015, accuracy=55.82%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 49.91%, Val acc: 55.82%

Epoch 38/250
Epoch 38/250 Training | loss: 0.018 | acc: 49.72% | lr: 5.30e-02  : 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10009/10009 [26:51<00:00,  6.21it/s]
Validating: 100%|████████████████████| 391/391 [00:41<00:00,  9.42it/s, loss=0.016, accuracy=53.00%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 49.72%, Val acc: 53.00%

Using device: cuda
/tmp/ipykernel_10287/523063070.py:35: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(
/tmp/ipykernel_10287/3148022116.py:17: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path)
Using 4 GPUs!

Loading checkpoint: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth
Resumed from epoch 38 with best accuracy: 57.30%

#increased batch size to 512
#increased base learning rate to 0.175

Epoch 39/250
Epoch 39/250 Training | loss: 0.003 | acc: 61.57% | lr: 5.35e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [18:02<00:00,  2.31it/s]
Validating: 100%|██████████████████████| 98/98 [00:53<00:00,  1.84it/s, loss=0.003, accuracy=66.26%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 61.57%, Val acc: 66.26%

Epoch 40/250
Epoch 40/250 Training | loss: 0.003 | acc: 62.83% | lr: 5.40e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:51<00:00,  2.33it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.30it/s, loss=0.003, accuracy=66.49%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 62.83%, Val acc: 66.49%

Epoch 41/250
Epoch 41/250 Training | loss: 0.003 | acc: 62.81% | lr: 5.45e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:48<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:45<00:00,  2.15it/s, loss=0.003, accuracy=65.89%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.81%, Val acc: 65.89%

Epoch 42/250
Epoch 42/250 Training | loss: 0.003 | acc: 62.57% | lr: 5.50e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:50<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:41<00:00,  2.36it/s, loss=0.003, accuracy=66.14%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.57%, Val acc: 66.14%

Epoch 43/250
Epoch 43/250 Training | loss: 0.003 | acc: 62.35% | lr: 5.55e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:48<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:24<00:00,  3.93it/s, loss=0.003, accuracy=63.88%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.35%, Val acc: 63.88%

Epoch 44/250
Epoch 44/250 Training | loss: 0.003 | acc: 62.23% | lr: 5.60e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:45<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:45<00:00,  2.16it/s, loss=0.003, accuracy=65.43%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.23%, Val acc: 65.43%

Epoch 45/250
Epoch 45/250 Training | loss: 0.003 | acc: 62.14% | lr: 5.65e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:50<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:24<00:00,  3.94it/s, loss=0.003, accuracy=65.33%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.14%, Val acc: 65.33%

Epoch 46/250
Epoch 46/250 Training | loss: 0.003 | acc: 62.12% | lr: 5.70e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:50<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.20it/s, loss=0.003, accuracy=64.97%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.12%, Val acc: 64.97%

Using device: cuda
/tmp/ipykernel_10287/1395723934.py:35: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(
/tmp/ipykernel_10287/3148022116.py:17: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path)
Using 4 GPUs!

Loading checkpoint: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth
Resumed from epoch 46 with best accuracy: 66.49%

Epoch 47/250
Epoch 47/250 Training | loss: 0.003 | acc: 62.14% | lr: 5.75e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:44<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:26<00:00,  3.66it/s, loss=0.003, accuracy=64.89%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.14%, Val acc: 64.89%

Epoch 48/250
Epoch 48/250 Training | loss: 0.003 | acc: 62.22% | lr: 5.80e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:46<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.14it/s, loss=0.003, accuracy=65.02%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.22%, Val acc: 65.02%

Epoch 49/250
Epoch 49/250 Training | loss: 0.003 | acc: 62.30% | lr: 5.85e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:47<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.21it/s, loss=0.003, accuracy=65.26%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.30%, Val acc: 65.26%

Epoch 50/250
Epoch 50/250 Training | loss: 0.003 | acc: 62.37% | lr: 5.90e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:45<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:24<00:00,  4.03it/s, loss=0.003, accuracy=64.96%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.37%, Val acc: 64.96%

Epoch 51/250
Epoch 51/250 Training | loss: 0.003 | acc: 62.37% | lr: 5.95e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:52<00:00,  2.33it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.20it/s, loss=0.003, accuracy=65.35%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.37%, Val acc: 65.35%

Epoch 52/250
Epoch 52/250 Training | loss: 0.003 | acc: 62.43% | lr: 6.00e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:50<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.33it/s, loss=0.003, accuracy=65.65%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.43%, Val acc: 65.65%

Epoch 53/250
Epoch 53/250 Training | loss: 0.003 | acc: 62.44% | lr: 6.05e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:47<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.21it/s, loss=0.003, accuracy=64.75%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.44%, Val acc: 64.75%

Epoch 54/250
Epoch 54/250 Training | loss: 0.003 | acc: 62.52% | lr: 6.10e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:49<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.35it/s, loss=0.003, accuracy=65.22%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.52%, Val acc: 65.22%

Epoch 55/250
Epoch 55/250 Training | loss: 0.003 | acc: 62.50% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:45<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.24it/s, loss=0.003, accuracy=63.65%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.50%, Val acc: 63.65%

#Removed onecycle policy and replaced with ReduceLROnPlateau scheduler on validation accuracy to reduce the learning rate when the validation accuracy plateaus

Using device: cuda
/tmp/ipykernel_10287/1001269296.py:35: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(
/tmp/ipykernel_10287/3148022116.py:17: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path)
Using 4 GPUs!

Loading checkpoint: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth
Resumed from epoch 55 with best accuracy: 66.49%

Epoch 56/250
Epoch 56/250 Training | loss: 0.003 | acc: 62.54% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:48<00:00,  2.34it/s]
Validating: 100%|██████████████████████| 98/98 [00:25<00:00,  3.85it/s, loss=0.003, accuracy=65.36%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.54%, Val acc: 65.36%

Epoch 57/250
Epoch 57/250 Training | loss: 0.003 | acc: 62.68% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:44<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.29it/s, loss=0.003, accuracy=64.34%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.68%, Val acc: 64.34%

Epoch 58/250
Epoch 58/250 Training | loss: 0.003 | acc: 62.65% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:45<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.25it/s, loss=0.003, accuracy=64.52%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.65%, Val acc: 64.52%

Epoch 59/250
Epoch 59/250 Training | loss: 0.003 | acc: 62.75% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.23it/s, loss=0.003, accuracy=65.37%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.75%, Val acc: 65.37%

Epoch 60/250
Epoch 60/250 Training | loss: 0.003 | acc: 62.84% | lr: 6.15e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:44<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.28it/s, loss=0.003, accuracy=65.31%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 62.84%, Val acc: 65.31%

Epoch 61/250
Epoch 61/250 Training | loss: 0.003 | acc: 67.42% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.27it/s, loss=0.002, accuracy=70.21%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 67.42%, Val acc: 70.21%

Epoch 62/250
Epoch 62/250 Training | loss: 0.003 | acc: 68.00% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:45<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.24it/s, loss=0.002, accuracy=70.60%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 68.00%, Val acc: 70.60%

Epoch 63/250
Epoch 63/250 Training | loss: 0.003 | acc: 68.00% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:44<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.28it/s, loss=0.002, accuracy=69.66%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 68.00%, Val acc: 69.66%

Epoch 64/250
Epoch 64/250 Training | loss: 0.003 | acc: 67.89% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.36it/s, loss=0.002, accuracy=69.71%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 67.89%, Val acc: 69.71%

Epoch 65/250
Epoch 65/250 Training | loss: 0.003 | acc: 67.81% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.35it/s, loss=0.002, accuracy=69.65%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 67.81%, Val acc: 69.65%

Epoch 66/250
Epoch 66/250 Training | loss: 0.003 | acc: 67.72% | lr: 3.07e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:23<00:00,  4.24it/s, loss=0.002, accuracy=69.26%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Results - Train acc: 67.72%, Val acc: 69.26%

Epoch 67/250
Epoch 67/250 Training | loss: 0.002 | acc: 71.07% | lr: 1.54e-02  : 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2502/2502 [17:43<00:00,  2.35it/s]
Validating: 100%|██████████████████████| 98/98 [00:22<00:00,  4.26it/s, loss=0.002, accuracy=72.82%]

Checkpoint saved: /home/ubuntu/checkpoints/resnet50_enhanced/checkpoint.pth

Best model saved: /home/ubuntu/checkpoints/resnet50_enhanced/best_model.pth

Results - Train acc: 71.07%, Val acc: 72.82%

Stopped after 67th epoch