     ┌────────────────────────────────────────────────────────────────────┐
     │                        • MobaXterm 20.3 •                          │
     │            (SSH client, X-server and networking tools)             │
     │                                                                    │
     │ ➤ SSH session to user10@430.guhk.cc                                │
     │   • SSH compression : ✔                                            │
     │   • SSH-browser     : ✔                                            │
     │   • X11-forwarding  : ✔  (remote display is forwarded through SSH) │
     │   • DISPLAY         : ✔  (automatically set on remote server)      │
     │                                                                    │
     │ ➤ For more info, ctrl+click on help or visit our website           │
     └────────────────────────────────────────────────────────────────────┘

Welcome to NVIDIA DGX Station Version 4.2.0 (GNU/Linux 4.15.0-55-generic x86_64)
Last login: Mon Feb  7 22:34:51 2022 from 111.187.79.12
user10@430:~$ conda activate pytorch
(pytorch) user10@430:~$ nvidia-smi
Tue Feb  8 16:34:37 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
| N/A   56C    P0    55W / 300W |   1289MiB / 32475MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
| N/A   66C    P0   243W / 300W |  21653MiB / 32478MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
| N/A   63C    P0   213W / 300W |  11267MiB / 32478MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
| N/A   62C    P0   253W / 300W |  14537MiB / 32478MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      3210      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      3480      G   /usr/bin/gnome-shell                          89MiB |
|    0      6936      C   python                                       953MiB |
|    0     24964      G   /usr/lib/xorg/Xorg                            84MiB |
|    0     25119      G   /usr/bin/gnome-shell                         129MiB |
|    1     23849      C   python                                     21641MiB |
|    2     23201      C   python                                     11255MiB |
|    3     22465      C   python                                     14525MiB |
+-----------------------------------------------------------------------------+
(pytorch) user10@430:~$ python main.py -device 0 -test0 True -real-size=2
python: can't open file 'main.py': [Errno 2] No such file or directory'
(pytorch) user10@430:~$ conda activate pytorch
(pytorch) user10@430:~$ cd ./lyx/src1
(pytorch) user10@430:~/lyx/src1$ python main.py -device 0 -test0 True -real-size=2
Using backend: pytorch
Loading data...
Parameters:
        BATCH_SIZE=32
        CANDIDATE_SIZE=10
        CATEGORY_NUM=500
        CLICK_SIZE=30
        CUDA=True
        DECAY=0.001
        DEVICE=0
        DROPOUT=0.5
        EARLY_STOPPING=1000
        EMBEDDING_DIM=198
        EMBEDDING_MID_DIM=48
        KERNEL_SIZE=3
        LOG_INTERVAL=1
        LR=0.0002
        MAX_NORM=3.0
        MULTICHANNEL=False
        NEWS_LIST_LENGTH=1025
        NODE_DROPOUT=0.1
        NON_STATIC=False
        NUM_ATTENTION_HEADS=16
        NUM_EPOCHS=10
        NUM_FILTERS=198
        NUM_WORDS_TITLE=100
        PRETRAINED_NAME=sgns.zhihu.word
        PRETRAINED_PATH=pretrained
        QUERY_VECTOR_DIM=128
        REAL_SIZE=2
        REFUSE_RATE=0.5
        REFUSE_SIZE=3
        SAVE_BEST=True
        SAVE_DIR=snapshot
        SNAPSHOT=None
        STATIC=False
        TEST0=True
        TEST_INTERVAL=100
        TESTZHIHU=False
        VOCAB_SIZE=250000
Traceback (most recent call last):
  File "main.py", line 126, in <module>
    model = (model0.MODEL0(args) if args.test0 else model.MODEL(args))
  File "/home/user10/lyx/src1/model0.py", line 18, in __init__
    self.user_encoder = UserEncoder(config)
  File "/home/user10/lyx/src1/model0.py", line 271, in __init__
    self.multihead_self_attention = MultiHeadSelfAttention(config.embedding_dim, config.num_attention_heads)
  File "/home/user10/lyx/src1/model0.py", line 374, in __init__
    assert d_model % num_attention_heads == 0
AssertionError
(pytorch) user10@430:~/lyx/src1$ python main.py -device 0 -test0 True -real-size=2
Using backend: pytorch
Loading data...
Parameters:
        BATCH_SIZE=32
        CANDIDATE_SIZE=10
        CATEGORY_NUM=500
        CLICK_SIZE=30
        CUDA=True
        DECAY=0.001
        DEVICE=0
        DROPOUT=0.5
        EARLY_STOPPING=1000
        EMBEDDING_DIM=256
        EMBEDDING_MID_DIM=48
        KERNEL_SIZE=3
        LOG_INTERVAL=1
        LR=0.0002
        MAX_NORM=3.0
        MULTICHANNEL=False
        NEWS_LIST_LENGTH=1025
        NODE_DROPOUT=0.1
        NON_STATIC=False
        NUM_ATTENTION_HEADS=16
        NUM_EPOCHS=10
        NUM_FILTERS=256
        NUM_WORDS_TITLE=100
        PRETRAINED_NAME=sgns.zhihu.word
        PRETRAINED_PATH=pretrained
        QUERY_VECTOR_DIM=128
        REAL_SIZE=2
        REFUSE_RATE=0.5
        REFUSE_SIZE=3
        SAVE_BEST=True
        SAVE_DIR=snapshot
        SNAPSHOT=None
        STATIC=False
        TEST0=True
        TEST_INTERVAL=100
        TESTZHIHU=False
        VOCAB_SIZE=250000
Batch[100] - loss: 395.549500  dcg: 0.6855 mrr: 0.6852 auc: 0.6074
Evaluation - loss: 490.246552  dcg: 59.3168 mrr: 48.9159 auc: 54.2969
Saving best model, acc: 59.3168%

Batch[200] - loss: 485.953644  dcg: 0.5487 mrr: 0.5266 auc: 0.3340 tensor([[[2.2991e-03, 8.8210e-01, 9.7924e-01,  ..., 7.0865e-01,
          4.4174e-02, 9.0957e-02],
         [1.6714e-02, 1.0563e+00, 1.4224e-01,  ..., 1.3017e+00,
          3.4566e-04, 1.2453e+00],
         [1.1238e-01, 1.1932e+00, 6.1182e-01,  ..., 6.7298e-01,
          2.1678e-03, 1.5183e-01],
         ...,
         [2.1185e-01, 2.1769e+00, 5.0075e-01,  ..., 1.0142e+00,
          3.7257e-01, 6.3543e-01],
         [1.0239e-01, 9.8700e-01, 1.0158e+00,  ..., 1.3162e+00,
          4.2674e-02, 5.7034e-01],
         [9.2980e-01, 7.5949e-01, 6.9400e-01,  ..., 9.7474e-01,
          8.4446e-02, 8.1214e-01]],

        [[4.2789e-01, 4.3510e-01, 4.2647e-01,  ..., 2.5092e-01,
          5.2074e-02, 9.1098e-01],
         [3.9244e-01, 6.0365e-03, 2.1522e-03,  ..., 1.7608e-02,
          6.4740e-03, 8.7530e-02],
         [2.6034e-01, 1.1555e+00, 1.0786e+00,  ..., 8.9643e-01,
          2.7344e-02, 2.2634e-01],
         ...,
         [2.4209e-01, 1.3558e+00, 8.8278e-01,  ..., 1.2974e+00,
          3.4440e-02, 2.4920e-01],
         [9.2755e-01, 9.5312e-01, 3.1743e-02,  ..., 9.3438e-01,
          2.9025e-02, 1.9217e-01],
         [5.8495e-01, 9.3354e-01, 4.3382e-01,  ..., 4.2358e-01,
          3.0232e-01, 4.1424e-01]],

        [[9.2947e-02, 2.1589e-01, 7.1553e-02,  ..., 3.5673e-01,
          2.2879e-01, 6.8041e-01],
         [9.0158e-01, 2.4452e+00, 7.4944e-01,  ..., 1.1660e+00,
          1.1739e-01, 1.1897e-01],
         [2.4751e-01, 5.5541e-02, 1.8667e-01,  ..., 4.8408e-01,
          2.6987e-01, 5.6695e-01],
         ...,
         [4.9610e-02, 1.0110e-01, 9.0222e-02,  ..., 4.9696e-01,
          1.1445e-01, 1.2734e+00],
         [4.2239e-03, 1.0843e+00, 6.4549e-01,  ..., 9.0346e-01,
          1.5386e-02, 2.1714e-01],
         [1.2213e+00, 7.3428e-01, 1.5250e-01,  ..., 7.8966e-02,
          1.4408e-01, 8.3147e-02]],

        ...,

        [[3.8601e-01, 9.5464e-04, 1.8509e-01,  ..., 8.8137e-03,
          3.1765e-03, 2.2537e+00],
         [1.2430e+00, 3.0093e-01, 7.7138e-02,  ..., 1.4730e+00,
          7.9703e-02, 1.4066e+00],
         [1.3313e-01, 3.6240e-02, 7.3412e-03,  ..., 1.9537e-01,
          3.3246e-01, 1.1380e-01],
         ...,
         [1.0137e-02, 4.9339e-01, 1.0451e+00,  ..., 4.3106e-02,
          3.3043e-01, 4.8976e-03],
         [3.0032e-02, 5.2822e-01, 1.1907e+00,  ..., 1.2590e+00,
          1.2291e-02, 4.1790e-01],
         [3.7884e-01, 5.9949e-01, 4.7910e-01,  ..., 7.5376e-01,
          5.2795e-02, 7.3800e-01]],

        [[3.9244e-01, 6.0365e-03, 2.1522e-03,  ..., 1.7608e-02,
          6.4740e-03, 8.7530e-02],
         [6.3245e-01, 6.0628e-01, 4.1756e-02,  ..., 2.3488e-01,
          2.2589e-01, 1.1167e+00],
         [4.3633e-01, 2.8984e-01, 2.0720e-02,  ..., 4.2897e-01,
          1.1350e-01, 3.8013e-01],
         ...,
         [9.8434e-02, 1.2586e+00, 3.9033e-01,  ..., 1.9694e+00,
          1.2300e-03, 2.6194e-01],
         [1.0137e-02, 4.9339e-01, 1.0451e+00,  ..., 4.3106e-02,
          3.3043e-01, 4.8976e-03],
         [8.8387e-01, 4.7852e-01, 4.3331e-01,  ..., 1.0367e+00,
          3.0545e-01, 4.7341e-01]],

        [[1.5382e-01, 1.3557e+00, 1.8076e+00,  ..., 1.6000e+00,
          2.9187e-01, 1.0631e+00],
         [3.6356e-02, 1.6677e-02, 1.5941e-01,  ..., 7.2066e-01,
          4.3418e-01, 1.6031e+00],
         [8.5354e-01, 7.9128e-01, 2.2421e-01,  ..., 1.7057e+00,
          5.7237e-01, 7.1149e-01],
         ...,
         [7.4634e-01, 2.5760e-02, 4.3682e-03,  ..., 1.0106e+00,
          1.8040e-02, 1.4906e+00],
         [1.0293e+00, 3.3378e-01, 4.1713e-02,  ..., 3.3021e-01,
          3.5906e-01, 7.9995e-01],
         [4.2018e-02, 2.5722e-02, 4.1369e-02,  ..., 2.9250e-02,
          2.9510e-02, 3.2705e-02]]], device='cuda:0', grad_fn=<CatBackward>) tensor([[2.2991e-03, 8.8210e-01, 9.7924e-01,  ..., 7.0865e-01, 4.4174e-02,
         9.0957e-02],
        [1.6714e-02, 1.0563e+00, 1.4224e-01,  ..., 1.3017e+00, 3.4566e-04,
         1.2453e+00],
        [1.1238e-01, 1.1932e+00, 6.1182e-01,  ..., 6.7298e-01, 2.1678e-03,
         1.5183e-01],
        ...,
        [2.1185e-01, 2.1769e+00, 5.0075e-01,  ..., 1.0142e+00, 3.7257e-01,
         6.3543e-01],
        [1.0239e-01, 9.8700e-01, 1.0158e+00,  ..., 1.3162e+00, 4.2674e-02,
         5.7034e-01],
        [9.2980e-01, 7.5949e-01, 6.9400e-01,  ..., 9.7474e-01, 8.4446e-02,
         8.1214e-01]], device='cuda:0', grad_fn=<SelectBackward>) tensor([2.2991e-03, 8.8210e-01, 9.7924e-01, 7.7569e-01, 6.1997e-01, 8.9796e-04,
        1.5777e-01, 6.2042e-01, 5.7049e-01, 4.4268e-04, 8.3907e-01, 2.2392e-02,
        4.2219e-02, 1.7473e+00, 9.9113e-01, 6.5656e-01, 7.5724e-03, 6.0330e-04,
        2.5192e+00, 2.9343e-03, 5.9700e-02, 2.7115e+00, 9.6651e-03, 2.0661e-01,
        3.0698e+00, 1.2632e+00, 3.9342e-01, 8.2976e-01, 3.0860e-02, 5.4420e-01,
        2.3807e-01, 7.6697e-01, 2.0365e+00, 3.4860e-01, 2.2372e-02, 2.6599e+00,
        1.5786e-02, 6.9639e-03, 5.7254e-01, 1.3118e-01, 6.6373e-01, 4.0796e-04,
        5.5776e-01, 5.6669e-02, 1.6255e+00, 5.8136e-01, 1.0014e-03, 9.7262e-04,
        7.1839e-01, 3.5970e-03, 2.4891e-02, 3.9603e-02, 1.8023e-02, 3.6127e+00,
        9.7260e-01, 2.0794e-02, 2.0630e-01, 4.6302e-02, 1.2142e-03, 3.2528e+00,
        1.7285e-01, 9.2781e-03, 4.3953e-01, 2.8241e+00, 9.4735e-06, 1.3118e-01,
        6.2120e-01, 5.0134e-01, 2.0159e+00, 4.4528e-01, 3.9632e-03, 5.2561e-01,
        1.5369e+00, 7.7165e-01, 1.0498e-03, 1.5811e-02, 1.1883e+00, 2.3029e-01,
        3.5393e-02, 4.1527e-03, 4.1181e-04, 4.6488e-02, 4.9451e-02, 2.1250e-02,
        1.1570e-02, 2.5151e-02, 2.8589e-02, 1.2767e-02, 1.8608e+00, 3.0639e-01,
        4.5626e-02, 3.8152e-01, 2.1195e-02, 2.3107e-01, 3.2549e-02, 7.8837e-04,
        1.8543e-01, 9.5245e-01, 3.0868e-02, 1.4519e-02, 5.1234e-01, 8.7126e-01,
        2.2742e+00, 1.4435e+00, 1.5524e-01, 3.2455e+00, 3.4360e-01, 1.3425e-03,
        1.6836e-02, 2.8948e+00, 5.4691e-02, 5.6840e-01, 2.2434e-02, 2.8019e-01,
        2.7286e+00, 1.0447e+00, 7.5907e-01, 1.1368e-01, 2.8285e-01, 1.4441e+00,
        8.8845e-01, 1.1684e+00, 1.2440e+00, 3.2990e+00, 1.5175e+00, 4.4432e-02,
        1.0276e+00, 5.4564e-01, 6.8543e-01, 9.6293e-01, 1.4458e+00, 4.6635e-01,
        1.4051e+00, 7.4031e-01, 1.9038e+00, 1.4165e+00, 1.4210e-03, 1.9850e-02,
        1.0567e-02, 8.4299e-01, 1.3383e+00, 4.5658e+00, 3.5350e-04, 2.8967e-03,
        1.3394e-02, 1.5777e-01, 1.8550e+00, 1.2820e-01, 7.5364e-01, 1.1100e-02,
        1.3168e+00, 1.2453e+00, 2.4232e-02, 1.1526e+00, 6.2696e-01, 2.9047e+00,
        2.3594e+00, 2.5516e-02, 5.9876e-04, 1.5972e-02, 3.9507e-02, 4.4475e+00,
        1.1134e-02, 7.2333e-04, 1.7229e-01, 2.6931e+00, 6.1783e-01, 2.5690e-03,
        4.5133e-01, 1.2584e-01, 3.3541e-01, 3.3907e-01, 4.6581e-01, 4.4077e-01,
        4.2503e-01, 4.5073e-01, 1.8924e+00, 7.7530e-01, 1.9345e+00, 9.4349e-03,
        4.9717e-02, 1.5027e-04, 3.5477e-02, 5.2787e-01, 2.7585e+00, 2.4966e-02,
        5.8694e-02, 3.1841e-02, 2.8045e+00, 1.1320e+00, 7.1281e-02, 4.6931e-04,
        7.8658e-01, 9.4777e-04, 1.3558e+00, 5.8812e-02, 5.4086e-02, 7.5168e-04,
        3.0975e-01, 2.3881e+00, 1.8586e+00, 3.7077e+00, 8.3598e-01, 1.2846e+00,
        2.0600e-01, 3.3588e+00, 4.5445e-04, 1.0404e-02, 3.1536e-02, 1.8154e-02,
        1.7925e+00, 3.2496e-03, 1.9058e-04, 9.8718e-01, 3.0864e-01, 2.5537e-01,
        2.7168e-05, 1.8295e+00, 1.0782e+00, 2.9934e-01, 5.5879e-01, 3.1249e+00,
        2.3198e+00, 1.0341e-01, 9.6996e-01, 6.1883e-02, 7.0488e-02, 1.2971e+00,
        1.3632e-03, 4.8781e-01, 2.9815e-01, 1.5154e+00, 5.1914e-05, 3.1786e+00,
        8.5391e-04, 3.4982e-04, 1.0333e+00, 4.7103e-02, 5.0554e-02, 2.6735e-01,
        2.2379e+00, 1.0773e+00, 7.5393e-01, 1.2612e-01, 4.3266e+00, 3.9811e-01,
        1.6328e-03, 1.6652e+00, 5.5940e-01, 6.0822e-02, 1.7859e+00, 7.5224e-03,
        1.2896e+00, 7.0865e-01, 4.4174e-02, 9.0957e-02], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([[0.6034, 0.7753, 0.5150,  ..., 0.8559, 0.1362, 0.6713],
        [0.4352, 0.7892, 0.4225,  ..., 0.9055, 0.2062, 0.8882],
        [0.5084, 0.6172, 0.3891,  ..., 0.7641, 0.1683, 0.6806],
        ...,
        [0.4525, 0.7727, 0.4649,  ..., 0.9704, 0.1102, 0.5695],
        [0.4944, 0.8272, 0.4486,  ..., 0.8644, 0.2023, 0.5343],
        [0.2491, 0.7055, 0.5467,  ..., 0.8088, 0.1927, 0.5714]],
       device='cuda:0', grad_fn=<SqueezeBackward1>) tensor([0.6034, 0.7753, 0.5150, 0.5326, 0.2348, 0.1065, 0.3107, 0.4193, 1.5834,
        0.4149, 0.8926, 0.4270, 0.1861, 1.8817, 1.6372, 1.1916, 0.0443, 0.9699,
        1.7250, 0.4829, 0.7448, 2.5789, 0.1336, 0.4396, 2.6662, 1.5099, 0.3553,
        0.9660, 0.1077, 0.5051, 0.2032, 1.3263, 1.2359, 0.4516, 0.0664, 2.2307,
        0.0929, 0.0424, 0.8213, 0.9189, 0.4569, 0.0748, 0.7128, 0.1788, 1.5007,
        1.3326, 0.2047, 0.0848, 0.5105, 0.3693, 0.1713, 0.1570, 0.4342, 2.5001,
        1.4524, 0.2783, 1.0209, 0.2281, 1.0281, 2.5698, 0.1409, 0.1943, 1.1278,
        2.6530, 0.2265, 0.0689, 0.4474, 0.6977, 2.8673, 0.1889, 0.3718, 0.9073,
        1.0363, 1.6819, 0.3691, 0.0687, 1.1568, 0.5242, 0.2047, 0.1058, 0.0425,
        0.3502, 1.0510, 0.2336, 0.0910, 0.1207, 0.3216, 0.2218, 1.5226, 0.4564,
        0.3517, 0.5207, 0.3584, 0.2724, 0.5401, 0.1442, 0.5452, 0.8936, 0.0991,
        0.2128, 1.0150, 0.8290, 1.9848, 1.5638, 0.2040, 1.8623, 0.7168, 0.5958,
        0.4653, 2.3400, 0.1510, 1.0160, 0.1499, 0.6100, 1.9830, 0.8685, 0.8463,
        0.1301, 0.0823, 1.5122, 1.8513, 0.7719, 1.6790, 2.3883, 1.9877, 0.1723,
        0.0731, 1.1099, 1.6100, 1.5740, 0.7306, 0.6233, 1.9114, 1.0578, 1.6421,
        0.6140, 0.1609, 0.4742, 0.2329, 0.7878, 1.5518, 3.7657, 0.1463, 0.0664,
        0.5978, 0.6157, 2.5455, 1.3450, 0.4256, 0.1951, 0.5374, 1.3930, 0.1194,
        1.2652, 0.5552, 2.2750, 1.6245, 0.5241, 0.0707, 0.0929, 0.2878, 3.1426,
        0.3414, 0.3434, 0.2632, 2.6561, 1.4910, 0.1823, 0.2940, 0.1091, 0.6910,
        0.3124, 0.1303, 0.5714, 0.5008, 0.2267, 1.3894, 1.4279, 2.0608, 0.2644,
        0.2660, 0.1696, 0.8193, 0.4047, 2.2858, 0.2617, 0.2727, 0.3087, 2.6001,
        1.3762, 0.0304, 0.4305, 1.2064, 0.4260, 1.7372, 0.8819, 0.1350, 0.0805,
        0.4118, 1.5911, 1.8039, 3.3437, 0.7741, 1.4792, 0.5029, 3.2490, 0.1183,
        0.7259, 0.2202, 0.2076, 1.7206, 0.5075, 0.2654, 1.6574, 0.9220, 0.7077,
        0.2921, 1.9920, 0.9027, 0.2304, 1.0391, 1.1981, 1.4864, 0.2746, 0.3787,
        0.5090, 0.2283, 0.9743, 0.0820, 0.3358, 0.7595, 1.6954, 0.1007, 2.6342,
        0.1218, 0.3175, 0.2534, 0.2294, 0.2612, 0.3001, 3.0886, 1.4572, 0.9219,
        0.2339, 2.8460, 0.4168, 0.4054, 1.6439, 1.3342, 0.4024, 1.8716, 0.7689,
        0.9260, 0.8559, 0.1362, 0.6713], device='cuda:0',
       grad_fn=<SelectBackward>)
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.9840e-24, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.9840e-24, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 3, 9, 8, 7, 6, 5, 4, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.9840e-24, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.8872e-05, 0.0000e+00, 0.0000e+00, 9.9998e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(1.8872e-05, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.8872e-05, 0.0000e+00, 0.0000e+00, 9.9998e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (4, 1, 9, 8, 7, 6, 5, 3, 2, 0) tensor([0.9200])
auc tensor([0.0000e+00, 1.8872e-05, 0.0000e+00, 0.0000e+00, 9.9998e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 8, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 3.1935e-31, 9.7481e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        2.5189e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(3.1935e-31, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 3.1935e-31, 9.7481e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        2.5189e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 6, 1, 9, 8, 7, 5, 4, 3, 0) tensor([0.7891])
auc tensor([0.0000e+00, 3.1935e-31, 9.7481e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        2.5189e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([5.1105e-42, 0.0000e+00, 0.0000e+00, 4.1488e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 5.8512e-01, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(5.1105e-42, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.1105e-42, 0.0000e+00, 0.0000e+00, 4.1488e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 5.8512e-01, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 3, 0, 9, 8, 6, 5, 4, 2, 1) tensor([0.7891])
auc tensor([5.1105e-42, 0.0000e+00, 0.0000e+00, 4.1488e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 5.8512e-01, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 8, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 5.1413e-22, 0.0000e+00, 2.8026e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(5.1413e-22, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 5.1413e-22, 0.0000e+00, 2.8026e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 1, 3, 9, 7, 6, 5, 4, 2, 0) tensor([0.9200])
auc tensor([0.0000e+00, 5.1413e-22, 0.0000e+00, 2.8026e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([0.0000, 0.7131, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2869,
        0.0000], device='cuda:0', grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.7131, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000, 0.7131, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2869,
        0.0000], device='cuda:0', grad_fn=<SelectBackward>) (1, 8, 9, 7, 6, 5, 4, 3, 2, 0) tensor([1.2891])
auc tensor([0.0000, 0.7131, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2869,
        0.0000], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.0000e+00, 3.6909e-23, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 3.6909e-23, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 9, 8, 7, 6, 5, 4, 3, 2) tensor([1.6309])
auc tensor([1.0000e+00, 3.6909e-23, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([5.7057e-40, 0.0000e+00, 2.7434e-36, 1.0000e+00, 2.1093e-34, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.5508e-22, 2.6209e-38], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([6], dtype=torch.int32) tensor(5.7057e-40, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.7057e-40, 0.0000e+00, 2.7434e-36, 1.0000e+00, 2.1093e-34, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.5508e-22, 2.6209e-38], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 8, 4, 2, 9, 0, 7, 6, 5, 1) tensor([0.6453])
auc tensor([5.7057e-40, 0.0000e+00, 2.7434e-36, 1.0000e+00, 2.1093e-34, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.5508e-22, 2.6209e-38], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.1875])
mrr tensor([5.0230e-21, 3.6427e-17, 4.8585e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        9.9951e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(3.6427e-17, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.0230e-21, 3.6427e-17, 4.8585e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        9.9951e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (6, 2, 1, 0, 9, 8, 7, 5, 4, 3) tensor([0.9307])
auc tensor([5.0230e-21, 3.6427e-17, 4.8585e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        9.9951e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6016e-35, 0.0000e+00, 0.0000e+00,
        2.8419e-13, 1.0000e+00, 0.0000e+00, 3.4633e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6016e-35, 0.0000e+00, 0.0000e+00,
        2.8419e-13, 1.0000e+00, 0.0000e+00, 3.4633e-25], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 6, 9, 3, 8, 5, 4, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6016e-35, 0.0000e+00, 0.0000e+00,
        2.8419e-13, 1.0000e+00, 0.0000e+00, 3.4633e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 7.5341e-19, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
        0.0000e+00, 1.6685e-38, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(7.5341e-19, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 7.5341e-19, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
        0.0000e+00, 1.6685e-38, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 1, 7, 9, 8, 6, 4, 3, 2, 0) tensor([0.9200])
auc tensor([0.0000e+00, 7.5341e-19, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
        0.0000e+00, 1.6685e-38, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([0.0000, 0.0000, 0.8606, 0.0237, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1157], device='cuda:0', grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000, 0.0000, 0.8606, 0.0237, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1157], device='cuda:0', grad_fn=<SelectBackward>) (2, 9, 3, 8, 7, 6, 5, 4, 1, 0) tensor([0.5901])
auc tensor([0.0000, 0.0000, 0.8606, 0.0237, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1157], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0504e-27, 3.0429e-34, 4.2039e-45,
        1.0000e+00, 0.0000e+00, 3.5633e-26, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([6], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0504e-27, 3.0429e-34, 4.2039e-45,
        1.0000e+00, 0.0000e+00, 3.5633e-26, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (6, 8, 3, 4, 5, 9, 7, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0504e-27, 3.0429e-34, 4.2039e-45,
        1.0000e+00, 0.0000e+00, 3.5633e-26, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 1.0000e+00, 4.0458e-27, 0.0000e+00, 0.0000e+00, 2.0191e-25,
        0.0000e+00, 1.6432e-30, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 4.0458e-27, 0.0000e+00, 0.0000e+00, 2.0191e-25,
        0.0000e+00, 1.6432e-30, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 5, 2, 7, 9, 8, 6, 4, 3, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 4.0458e-27, 0.0000e+00, 0.0000e+00, 2.0191e-25,
        0.0000e+00, 1.6432e-30, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.5414e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.7311e-36, 4.2534e-34, 5.8379e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.5414e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.7311e-36, 4.2534e-34, 5.8379e-25], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 9, 8, 7, 3, 6, 5, 4, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.5414e-44, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.7311e-36, 4.2534e-34, 5.8379e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.0000e+00, 7.0373e-35, 0.0000e+00, 2.1946e-36, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 7.0373e-35, 0.0000e+00, 2.1946e-36, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 3, 9, 8, 7, 6, 5, 4, 2) tensor([1.6309])
auc tensor([1.0000e+00, 7.0373e-35, 0.0000e+00, 2.1946e-36, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 8, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 0.0000e+00, 5.0767e-36, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 5.0767e-36, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 2, 9, 8, 6, 5, 4, 3, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 5.0767e-36, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([8.4516e-10, 0.0000e+00, 3.7067e-03, 0.0000e+00, 9.9629e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 3.2277e-09, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(8.4516e-10, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([8.4516e-10, 0.0000e+00, 3.7067e-03, 0.0000e+00, 9.9629e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 3.2277e-09, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (4, 2, 8, 0, 9, 7, 6, 5, 3, 1) tensor([0.7197])
auc tensor([8.4516e-10, 0.0000e+00, 3.7067e-03, 0.0000e+00, 9.9629e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 3.2277e-09, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3125])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 4.4654e-32, 0.0000e+00, 0.0000e+00,
        8.1047e-19, 0.0000e+00, 0.0000e+00, 7.4539e-30], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 4.4654e-32, 0.0000e+00, 0.0000e+00,
        8.1047e-19, 0.0000e+00, 0.0000e+00, 7.4539e-30], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 9, 3, 8, 7, 5, 4, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 4.4654e-32, 0.0000e+00, 0.0000e+00,
        8.1047e-19, 0.0000e+00, 0.0000e+00, 7.4539e-30], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([2.1230e-02, 9.7877e-01, 0.0000e+00, 0.0000e+00, 2.3202e-36, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4106e-17], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9788, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.1230e-02, 9.7877e-01, 0.0000e+00, 0.0000e+00, 2.3202e-36, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4106e-17], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 9, 4, 8, 7, 6, 5, 3, 2) tensor([1.6309])
auc tensor([2.1230e-02, 9.7877e-01, 0.0000e+00, 0.0000e+00, 2.3202e-36, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4106e-17], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 8.0719e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 8.0719e-14], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 9, 2, 8, 7, 6, 5, 4, 3, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 8.0719e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 9.1566e-03, 0.0000e+00, 0.0000e+00, 3.0486e-15, 1.1876e-13,
        6.0363e-22, 9.9084e-01, 8.7060e-26, 2.1651e-31], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.0092, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 9.1566e-03, 0.0000e+00, 0.0000e+00, 3.0486e-15, 1.1876e-13,
        6.0363e-22, 9.9084e-01, 8.7060e-26, 2.1651e-31], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 1, 5, 4, 6, 8, 9, 3, 2, 0) tensor([0.9200])
auc tensor([0.0000e+00, 9.1566e-03, 0.0000e+00, 0.0000e+00, 3.0486e-15, 1.1876e-13,
        6.0363e-22, 9.9084e-01, 8.7060e-26, 2.1651e-31], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([9.8603e-04, 6.8086e-24, 9.9901e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.0010, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.8603e-04, 6.8086e-24, 9.9901e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 0, 1, 9, 8, 7, 6, 5, 4, 3) tensor([1.1309])
auc tensor([9.8603e-04, 6.8086e-24, 9.9901e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8750])
mrr tensor([0.0000e+00, 1.5577e-33, 7.3232e-16, 5.1750e-23, 1.0000e+00, 0.0000e+00,
        5.2401e-25, 0.0000e+00, 0.0000e+00, 8.2593e-40], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(1.5577e-33, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.5577e-33, 7.3232e-16, 5.1750e-23, 1.0000e+00, 0.0000e+00,
        5.2401e-25, 0.0000e+00, 0.0000e+00, 8.2593e-40], device='cuda:0',
       grad_fn=<SelectBackward>) (4, 2, 3, 6, 1, 9, 8, 7, 5, 0) tensor([0.6759])
auc tensor([0.0000e+00, 1.5577e-33, 7.3232e-16, 5.1750e-23, 1.0000e+00, 0.0000e+00,
        5.2401e-25, 0.0000e+00, 0.0000e+00, 8.2593e-40], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.2500])
mrr tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.7504e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.7504e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 8, 9, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.7504e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 9.9560e-32, 9.6945e-12,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 9.9560e-32, 9.6945e-12,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 5, 4, 9, 8, 7, 6, 3, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 9.9560e-32, 9.6945e-12,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1034e-13, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1034e-13, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 3, 9, 8, 6, 5, 4, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1034e-13, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 2.8091e-08, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.9718e-21, 0.0000e+00, 0.0000e+00, 1.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 2.8091e-08, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.9718e-21, 0.0000e+00, 0.0000e+00, 1.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (9, 2, 6, 8, 7, 5, 4, 3, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 2.8091e-08, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.9718e-21, 0.0000e+00, 0.0000e+00, 1.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])

Evaluation - loss: 481.511475  dcg: 55.8390 mrr: 53.5941 auc: 34.4758
Batch[300] - loss: 505.999603  dcg: 0.5205 mrr: 0.5427 auc: 0.2676
Evaluation - loss: 483.837311  dcg: 53.8796 mrr: 56.9502 auc: 29.2717
Batch[400] - loss: 436.000000  dcg: 0.5710 mrr: 0.6443 auc: 0.2793
Evaluation - loss: 478.856049  dcg: 52.7038 mrr: 59.7071 auc: 24.2125
Batch[500] - loss: 486.000000  dcg: 0.5026 mrr: 0.5807 auc: 0.1992
Evaluation - loss: 474.634338  dcg: 52.1180 mrr: 61.0274 auc: 21.7805
Batch[600] - loss: 495.997742  dcg: 0.5326 mrr: 0.5083 auc: 0.3027 tensor([[[5.2040e-01, 6.5185e-02, 1.5914e-01,  ..., 1.1299e+00,
          5.8820e-01, 1.9994e+00],
         [1.6422e-01, 7.2651e-01, 3.6469e-01,  ..., 1.0067e+00,
          3.4934e-01, 2.2036e+00],
         [7.2934e-01, 6.5456e-01, 4.5707e-01,  ..., 2.4623e+00,
          4.6439e-01, 4.1202e+00],
         ...,
         [7.6419e-01, 5.1450e-01, 4.8382e-01,  ..., 1.4491e+00,
          7.7516e-01, 1.6748e+00],
         [1.7144e-01, 1.4603e-01, 5.7496e-01,  ..., 1.8055e+00,
          4.0770e-01, 2.7116e+00],
         [6.6700e-01, 6.1089e-01, 4.1005e-01,  ..., 1.5502e+00,
          1.0622e+00, 2.4689e+00]],

        [[3.5176e-01, 3.5162e-01, 1.9513e-01,  ..., 1.5956e+00,
          3.8704e-01, 1.7777e+00],
         [6.3587e-01, 4.3335e-01, 6.9741e-01,  ..., 2.4656e+00,
          8.4600e-01, 4.0711e+00],
         [4.5901e-01, 3.7386e-01, 7.2353e-01,  ..., 1.4764e+00,
          9.3282e-01, 2.6122e+00],
         ...,
         [2.1540e-01, 2.2988e-01, 3.8296e-01,  ..., 9.9562e-01,
          3.9705e-01, 2.3077e+00],
         [2.7167e-01, 4.1202e-01, 6.4473e-01,  ..., 2.6774e+00,
          6.7962e-01, 3.9785e+00],
         [9.4103e-02, 9.2412e-02, 2.5779e-01,  ..., 3.7091e-01,
          8.2762e-02, 1.4918e+00]],

        [[1.5476e-01, 3.1267e-01, 4.2185e-01,  ..., 1.5876e+00,
          1.0756e-01, 2.9277e+00],
         [4.1519e-01, 1.5374e-01, 5.8226e-01,  ..., 2.4965e+00,
          6.7880e-01, 4.1052e+00],
         [2.7098e-01, 2.6989e-01, 2.2614e-01,  ..., 5.9434e-01,
          2.2081e-01, 2.8974e+00],
         ...,
         [6.2367e-01, 2.4977e-01, 4.5782e-01,  ..., 2.3421e+00,
          5.0489e-01, 3.0678e+00],
         [2.2146e-01, 3.4182e-01, 3.1128e-01,  ..., 1.6207e+00,
          5.9962e-01, 2.6991e+00],
         [7.6742e-02, 4.0775e-02, 1.5921e-01,  ..., 4.6477e-01,
          1.9573e-01, 1.5322e+00]],

        ...,

        [[2.8894e-01, 4.0441e-01, 4.0857e-01,  ..., 1.0159e+00,
          1.0069e-01, 2.1617e+00],
         [4.7850e-01, 3.4656e-01, 3.3688e-01,  ..., 2.9744e+00,
          2.6560e-01, 4.5716e+00],
         [7.0643e-02, 1.1867e-01, 1.5045e-01,  ..., 4.9940e-01,
          1.8949e-03, 6.2026e-01],
         ...,
         [1.8487e-01, 2.5752e-02, 2.8924e-02,  ..., 1.0115e+00,
          5.4137e-02, 1.2393e+00],
         [6.0511e-01, 2.9855e-01, 6.7415e-01,  ..., 1.2445e+00,
          3.4231e-01, 2.3651e+00],
         [4.3478e-01, 2.1659e-01, 3.6384e-01,  ..., 2.1944e+00,
          3.1366e-01, 3.3180e+00]],

        [[3.5223e-01, 4.1709e-01, 4.5274e-01,  ..., 2.0496e+00,
          6.1928e-01, 3.5834e+00],
         [5.4620e-01, 1.0326e+00, 8.8063e-01,  ..., 1.6513e+00,
          8.1088e-01, 2.6767e+00],
         [1.5476e-01, 3.1267e-01, 4.2185e-01,  ..., 1.5876e+00,
          1.0756e-01, 2.9277e+00],
         ...,
         [4.6498e-01, 5.8368e-01, 5.2631e-01,  ..., 2.4631e+00,
          8.0996e-01, 3.9266e+00],
         [5.4335e-01, 1.1155e-01, 3.9334e-01,  ..., 1.4061e+00,
          2.5714e-01, 2.8836e+00],
         [7.5999e-01, 3.0970e-01, 7.8147e-01,  ..., 1.8230e+00,
          4.5661e-01, 2.2619e+00]],

        [[7.3118e-02, 9.0256e-02, 3.2284e-01,  ..., 6.1831e-01,
          1.5055e-01, 9.5879e-01],
         [5.3675e-01, 7.9447e-01, 4.6388e-01,  ..., 2.1050e+00,
          2.7091e-01, 4.0650e+00],
         [6.3384e-01, 1.4710e-01, 3.4531e-01,  ..., 1.5509e+00,
          6.1697e-01, 3.2007e+00],
         ...,
         [2.4480e-01, 2.8762e-01, 4.4080e-01,  ..., 1.0941e+00,
          1.5056e-01, 1.4340e+00],
         [9.4278e-01, 4.4713e-01, 6.6264e-01,  ..., 2.3609e+00,
          9.8146e-01, 4.5663e+00],
         [1.5147e-01, 3.1196e-01, 4.0253e-01,  ..., 1.0896e+00,
          3.0998e-01, 2.8863e+00]]], device='cuda:0', grad_fn=<CatBackward>) tensor([[0.5204, 0.0652, 0.1591,  ..., 1.1299, 0.5882, 1.9994],
        [0.1642, 0.7265, 0.3647,  ..., 1.0067, 0.3493, 2.2036],
        [0.7293, 0.6546, 0.4571,  ..., 2.4623, 0.4644, 4.1202],
        ...,
        [0.7642, 0.5145, 0.4838,  ..., 1.4491, 0.7752, 1.6748],
        [0.1714, 0.1460, 0.5750,  ..., 1.8055, 0.4077, 2.7116],
        [0.6670, 0.6109, 0.4101,  ..., 1.5502, 1.0622, 2.4689]],
       device='cuda:0', grad_fn=<SelectBackward>) tensor([5.2040e-01, 6.5185e-02, 1.5914e-01, 3.9169e-01, 2.8259e-01, 3.0348e-01,
        2.1679e-01, 4.2247e-01, 9.6501e-01, 3.9496e-01, 1.7064e+00, 1.2033e+00,
        2.9744e-01, 1.5612e+00, 1.6164e+00, 9.1639e-01, 3.1441e-01, 4.5354e-01,
        8.3831e-01, 6.1067e-01, 8.7338e-01, 1.1950e+00, 3.3914e-01, 2.9915e-01,
        1.9614e+00, 1.5931e+00, 5.0394e-01, 1.1608e+00, 3.1915e-01, 1.1293e-01,
        6.0223e-01, 8.9829e-01, 9.7193e-01, 1.1349e+00, 2.2261e-01, 1.6517e+00,
        3.6695e-01, 2.5523e-01, 2.9800e-01, 1.9523e+00, 1.1117e+00, 3.2981e-01,
        1.3470e+00, 1.8503e-01, 7.6737e-01, 1.2224e+00, 1.6867e-01, 3.5371e-01,
        7.1073e-01, 7.4914e-01, 2.7149e-01, 8.1577e-02, 3.1518e-01, 2.6910e+00,
        1.8479e+00, 1.2846e-01, 1.1433e+00, 1.5310e-01, 1.0262e+00, 1.9204e+00,
        4.6306e-01, 1.9051e-01, 1.3891e+00, 1.5995e+00, 6.5499e-01, 1.1125e-01,
        3.7731e-01, 7.8767e-01, 1.9366e+00, 4.3090e-01, 3.9462e-01, 1.0129e+00,
        4.3225e-01, 1.8772e+00, 6.2530e-01, 3.9206e-02, 1.2044e+00, 4.1899e-01,
        3.3392e-01, 4.3132e-01, 2.5501e-01, 1.0187e+00, 1.9319e+00, 1.3570e+00,
        2.3649e-01, 7.2762e-01, 3.3953e-01, 1.2439e+00, 1.7884e+00, 2.9285e-01,
        7.1977e-01, 3.0179e-01, 6.6884e-01, 1.6362e-01, 3.6254e-01, 1.7460e-01,
        5.0943e-01, 1.5081e+00, 1.2030e-01, 1.2208e-01, 9.9320e-01, 1.0237e+00,
        1.5348e+00, 1.4320e+00, 4.6760e-01, 1.5424e+00, 7.4792e-01, 6.3858e-01,
        6.8351e-01, 1.4482e+00, 4.4687e-01, 6.9804e-01, 4.7427e-01, 1.4521e+00,
        9.3244e-01, 8.1590e-01, 9.5245e-01, 3.5897e-01, 4.0419e-01, 1.6893e+00,
        1.7971e+00, 2.5153e-01, 1.8376e+00, 1.8820e+00, 1.3835e+00, 4.1500e-01,
        2.2428e-01, 5.7963e-01, 2.1146e+00, 1.5672e+00, 7.3380e-01, 3.0421e-01,
        1.7818e+00, 1.6208e+00, 1.6014e+00, 1.6553e-01, 2.6311e-01, 1.0056e-01,
        4.0783e-01, 4.1308e-01, 2.0739e+00, 2.0603e+00, 1.1484e+00, 3.9414e-01,
        7.1860e-01, 4.3016e-01, 1.9696e+00, 1.5142e+00, 5.9065e-01, 5.6927e-01,
        1.5566e-01, 1.3894e+00, 2.7157e-01, 2.2691e+00, 6.7552e-01, 2.5159e+00,
        1.3427e+00, 5.7907e-01, 2.5924e-01, 3.7990e-01, 3.1062e-01, 1.7283e+00,
        3.4902e-01, 6.8542e-01, 6.3862e-01, 2.3630e+00, 7.5020e-01, 8.9334e-02,
        3.3238e-01, 2.7490e-01, 1.0963e+00, 1.1511e-01, 2.7286e-01, 2.3521e-01,
        2.8004e-01, 6.9856e-04, 1.5755e+00, 1.8905e+00, 1.7694e+00, 3.7601e-01,
        1.0266e+00, 2.1103e-01, 1.7416e+00, 6.9737e-01, 1.7495e+00, 4.0378e-01,
        3.6967e-01, 2.4006e-01, 1.8090e+00, 4.6797e-01, 3.1547e-01, 3.4146e-01,
        9.8555e-01, 3.4286e-01, 1.1846e+00, 1.6039e+00, 4.3295e-01, 3.1787e-01,
        2.4227e-01, 1.3023e+00, 1.9570e+00, 2.4739e+00, 5.2610e-01, 2.1879e+00,
        3.9457e-01, 2.3527e+00, 7.6254e-01, 2.8170e-01, 5.1358e-01, 8.2468e-01,
        9.4615e-01, 5.7085e-01, 4.4776e-01, 2.0614e+00, 1.4537e+00, 1.1892e+00,
        7.3752e-01, 1.4234e+00, 6.9132e-01, 5.2829e-01, 9.3821e-01, 1.8235e+00,
        1.2120e+00, 3.6874e-01, 4.9494e-01, 4.9495e-01, 3.4839e-01, 8.9802e-01,
        2.0855e-01, 6.7301e-02, 9.9155e-01, 1.3326e+00, 2.4647e-01, 2.4096e+00,
        3.0348e-01, 3.8607e-01, 6.1119e-02, 4.4006e-01, 2.7238e-01, 2.7052e-01,
        1.7053e+00, 1.2068e+00, 7.8619e-01, 2.9274e-01, 1.9067e+00, 1.4419e-01,
        3.3941e-01, 1.2004e+00, 1.3335e+00, 7.7688e-01, 1.7300e+00, 1.0853e+00,
        8.3237e-01, 1.1299e+00, 5.8820e-01, 1.9994e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([[0.4524, 0.3489, 0.4443,  ..., 1.5193, 0.4163, 2.4702],
        [0.4525, 0.3252, 0.4180,  ..., 1.7046, 0.4309, 2.8068],
        [0.5456, 0.3047, 0.4160,  ..., 1.9746, 0.4671, 3.1913],
        ...,
        [0.3633, 0.3099, 0.3718,  ..., 1.4338, 0.3065, 2.5221],
        [0.4002, 0.3145, 0.4403,  ..., 1.6090, 0.4226, 2.7347],
        [0.4556, 0.3425, 0.3887,  ..., 1.6058, 0.4013, 2.7005]],
       device='cuda:0', grad_fn=<SqueezeBackward1>) tensor([0.4524, 0.3489, 0.4443, 0.4910, 0.3750, 0.3622, 0.4791, 0.4828, 1.6050,
        0.5028, 1.4272, 1.0558, 0.4138, 1.6269, 1.5586, 1.1872, 0.3520, 0.4797,
        1.5034, 1.1077, 0.9927, 1.2374, 0.3480, 0.3869, 2.1809, 1.7383, 0.7248,
        1.3705, 0.4612, 1.0120, 1.0570, 0.7467, 1.6197, 1.3050, 0.3722, 2.4231,
        0.3766, 0.3324, 0.8357, 1.7850, 1.6838, 0.3369, 1.6769, 0.4588, 0.8591,
        1.0224, 0.3457, 0.4287, 0.7340, 0.7246, 0.4338, 0.6402, 0.6552, 2.4572,
        1.4479, 0.4950, 0.8867, 0.4112, 0.9408, 2.1440, 0.8925, 0.4577, 1.3572,
        2.4550, 0.8325, 0.3252, 0.4152, 1.1105, 2.2612, 0.5135, 0.6898, 1.3254,
        0.4889, 2.3224, 0.7490, 0.3061, 0.6750, 0.6130, 0.4939, 0.4064, 0.3823,
        0.6316, 1.6411, 1.6118, 0.5171, 1.0428, 0.3632, 1.5113, 1.7468, 0.4880,
        0.8277, 0.5391, 0.4232, 0.2634, 0.6989, 0.3670, 0.4110, 1.0896, 0.3895,
        0.4695, 1.3263, 1.6819, 1.5938, 1.6114, 0.6688, 1.2608, 0.7077, 0.7004,
        0.5534, 1.9953, 0.4698, 0.9542, 0.5595, 1.0741, 0.8898, 1.4116, 1.4612,
        0.3649, 0.3347, 1.8510, 1.7318, 0.8072, 2.3329, 2.3642, 1.2204, 0.4338,
        0.4293, 0.4894, 1.4519, 1.8797, 0.8515, 0.7088, 2.0496, 1.6958, 1.3730,
        0.6391, 0.6037, 0.3649, 0.4223, 0.6422, 1.7857, 2.8115, 1.3133, 0.3624,
        0.7725, 0.6110, 2.4388, 1.3559, 0.4940, 0.9647, 0.5562, 1.2891, 0.3814,
        2.0512, 0.7990, 2.4418, 1.6130, 0.6472, 0.8199, 0.6266, 0.3603, 2.4780,
        0.5612, 0.7213, 0.4694, 2.0752, 0.6154, 0.3470, 0.3476, 0.4233, 1.2913,
        0.4165, 0.4496, 0.4798, 0.5688, 0.3969, 1.8332, 1.7296, 1.9361, 0.4528,
        1.0432, 0.3375, 1.6925, 0.5764, 1.3287, 0.5288, 0.5310, 0.3125, 2.0034,
        0.8241, 0.1817, 0.3518, 1.1496, 0.5070, 1.4199, 1.6661, 0.3624, 0.6159,
        0.6574, 2.2367, 1.8452, 2.3192, 1.2077, 2.5028, 0.3591, 2.5396, 0.6316,
        0.4901, 0.5106, 0.6144, 1.5488, 0.8567, 0.5791, 2.2834, 1.1818, 0.9776,
        0.4177, 2.2221, 1.0391, 0.4903, 1.3691, 1.4771, 1.2436, 0.3987, 0.4842,
        0.5283, 0.2654, 1.0518, 0.7682, 0.2718, 1.3234, 1.5685, 0.3423, 2.3000,
        0.4634, 0.4220, 0.2641, 0.2804, 0.5380, 0.4831, 1.8263, 1.6431, 0.9416,
        0.4501, 2.6360, 0.6393, 0.4542, 2.1067, 1.1560, 0.9603, 2.3123, 0.8019,
        1.3435, 1.5193, 0.4163, 2.4702], device='cuda:0',
       grad_fn=<SelectBackward>)
mrr tensor([0.0000e+00, 0.0000e+00, 2.0469e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        2.7594e-25, 0.0000e+00, 1.4726e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 2.0469e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        2.7594e-25, 0.0000e+00, 1.4726e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 2, 6, 8, 9, 7, 5, 4, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 2.0469e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        2.7594e-25, 0.0000e+00, 1.4726e-31, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.4661e-32, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.4661e-32, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 6, 9, 7, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.4661e-32, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 2.4123e-20, 0.0000e+00, 2.2632e-41, 3.0427e-17, 0.0000e+00,
        0.0000e+00, 6.9631e-23, 1.0000e+00, 4.5918e-24], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(2.4123e-20, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 2.4123e-20, 0.0000e+00, 2.2632e-41, 3.0427e-17, 0.0000e+00,
        0.0000e+00, 6.9631e-23, 1.0000e+00, 4.5918e-24], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 4, 1, 7, 9, 3, 6, 5, 2, 0) tensor([0.7891])
auc tensor([0.0000e+00, 2.4123e-20, 0.0000e+00, 2.2632e-41, 3.0427e-17, 0.0000e+00,
        0.0000e+00, 6.9631e-23, 1.0000e+00, 4.5918e-24], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 8, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 1.3669e-21, 5.4372e-41, 0.0000e+00,
        0.0000e+00, 1.9436e-42, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 1.3669e-21, 5.4372e-41, 0.0000e+00,
        0.0000e+00, 1.9436e-42, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 3, 4, 7, 9, 8, 6, 5, 2, 1) tensor([1.2891])
auc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 1.3669e-21, 5.4372e-41, 0.0000e+00,
        0.0000e+00, 1.9436e-42, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.6494e-33, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(1.6494e-33, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.6494e-33, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 1, 9, 7, 6, 5, 4, 3, 2, 0) tensor([0.9200])
auc tensor([0.0000e+00, 1.6494e-33, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([0.0000e+00, 1.2029e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.5925e-07, 1.7317e-24, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(1.2029e-19, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.2029e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.5925e-07, 1.7317e-24, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 7, 1, 8, 9, 6, 5, 4, 3, 0) tensor([0.7891])
auc tensor([0.0000e+00, 1.2029e-19, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.5925e-07, 1.7317e-24, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([2.6388e-12, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(2.6388e-12, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.6388e-12, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 0, 9, 8, 7, 6, 5, 4, 2, 1) tensor([0.9200])
auc tensor([2.6388e-12, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 9, 8, 7, 6, 5, 4, 3, 2, 0) tensor([1.2891])
auc tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.7526e-14, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.7526e-14, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 9, 8, 7, 5, 4, 3, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.7526e-14, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.4006e-06, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1.0000, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.4006e-06, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 3, 9, 8, 7, 6, 5, 4, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 1.4006e-06, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4013e-45], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4013e-45], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 3, 9, 8, 7, 6, 5, 4, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 9.8091e-45, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4013e-45], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 4.2186e-35,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 4.2186e-35,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 5, 9, 8, 7, 6, 4, 3, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 4.2186e-35,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9975e-15, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9975e-15, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 4, 9, 8, 6, 5, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9975e-15, 0.0000e+00,
        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 9, 7, 6, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 9, 8, 6, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([9.3743e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.6843e-10,
        6.2567e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9374, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.3743e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.6843e-10,
        6.2567e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 6, 5, 9, 8, 7, 4, 3, 2, 1) tensor([1.2891])
auc tensor([9.3743e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.6843e-10,
        6.2567e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.0000e+00, 2.6984e-16, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 2.6984e-16, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 9, 8, 7, 6, 5, 4, 3, 2) tensor([1.6309])
auc tensor([1.0000e+00, 2.6984e-16, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([0.0000e+00, 6.6781e-19, 2.3873e-11, 1.0000e+00, 0.0000e+00, 2.1487e-16,
        1.4013e-45, 2.3854e-38, 1.0117e-38, 1.0261e-27], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(6.6781e-19, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 6.6781e-19, 2.3873e-11, 1.0000e+00, 0.0000e+00, 2.1487e-16,
        1.4013e-45, 2.3854e-38, 1.0117e-38, 1.0261e-27], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 2, 5, 1, 9, 7, 8, 6, 4, 0) tensor([0.7197])
auc tensor([0.0000e+00, 6.6781e-19, 2.3873e-11, 1.0000e+00, 0.0000e+00, 2.1487e-16,
        1.4013e-45, 2.3854e-38, 1.0117e-38, 1.0261e-27], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3125])
mrr tensor([0.0000e+00, 2.4756e-29, 0.0000e+00, 3.2913e-11, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(2.4756e-29, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 2.4756e-29, 0.0000e+00, 3.2913e-11, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 3, 1, 9, 7, 6, 5, 4, 2, 0) tensor([0.7891])
auc tensor([0.0000e+00, 2.4756e-29, 0.0000e+00, 3.2913e-11, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 9, 8, 7, 6, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 9, 7, 6, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.2191e-43, 2.0604e-28, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.2191e-43, 2.0604e-28, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 7, 6, 9, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.2191e-43, 2.0604e-28, 1.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 9, 8, 7, 6, 5, 4, 3, 1, 0) tensor([0.5901])
auc tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 9, 8, 7, 6, 5, 4, 3, 2, 0) tensor([1.2891])
auc tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9736e-42, 0.0000e+00,
        0.0000e+00, 1.1073e-15, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9736e-42, 0.0000e+00,
        0.0000e+00, 1.1073e-15, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 7, 4, 9, 8, 6, 5, 3, 2, 1) tensor([1.2891])
auc tensor([1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9736e-42, 0.0000e+00,
        0.0000e+00, 1.1073e-15, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 4.0571e-32, 0.0000e+00, 0.0000e+00, 1.1659e-04, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 9.9988e-01, 8.1116e-19], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(4.0571e-32, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 4.0571e-32, 0.0000e+00, 0.0000e+00, 1.1659e-04, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 9.9988e-01, 8.1116e-19], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 4, 9, 1, 7, 6, 5, 3, 2, 0) tensor([0.7197])
auc tensor([0.0000e+00, 4.0571e-32, 0.0000e+00, 0.0000e+00, 1.1659e-04, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 9.9988e-01, 8.1116e-19], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3125])
mrr tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 8, 7, 6, 5, 4, 3, 2, 1) tensor([1.2891])
auc tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 7.2435e-18, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 7.2435e-18, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 7, 9, 8, 6, 5, 4, 3, 2, 0) tensor([1.2891])
auc tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 7.2435e-18, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([1.4020e-16, 9.9919e-01, 0.0000e+00, 0.0000e+00, 8.0876e-04, 0.0000e+00,
        0.0000e+00, 1.4013e-45, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9992, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.4020e-16, 9.9919e-01, 0.0000e+00, 0.0000e+00, 8.0876e-04, 0.0000e+00,
        0.0000e+00, 1.4013e-45, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 4, 0, 7, 9, 8, 6, 5, 3, 2) tensor([1.5000])
auc tensor([1.4020e-16, 9.9919e-01, 0.0000e+00, 0.0000e+00, 8.0876e-04, 0.0000e+00,
        0.0000e+00, 1.4013e-45, 0.0000e+00, 0.0000e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.9375])
mrr tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.0000e+00, 4.9309e-18, 0.0000e+00, 5.3740e-37], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.0000e+00, 4.9309e-18, 0.0000e+00, 5.3740e-37], device='cuda:0',
       grad_fn=<SelectBackward>) (6, 7, 9, 8, 5, 4, 3, 2, 1, 0) tensor([0.5901])
auc tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.0000e+00, 4.9309e-18, 0.0000e+00, 5.3740e-37], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])

Evaluation - loss: 483.136353  dcg: 54.3669 mrr: 54.7164 auc: 30.5570
Batch[700] - loss: 468.106812  dcg: 0.5740 mrr: 0.5347 auc: 0.3848
Evaluation - loss: 477.136230  dcg: 58.1523 mrr: 51.5466 auc: 43.8508
Batch[800] - loss: 465.977173  dcg: 0.5745 mrr: 0.5646 auc: 0.3750
Evaluation - loss: 480.885773  dcg: 56.0097 mrr: 52.5260 auc: 35.4272
Batch[900] - loss: 482.140381  dcg: 0.5455 mrr: 0.5086 auc: 0.3145
Evaluation - loss: 478.193481  dcg: 54.5950 mrr: 56.6703 auc: 28.9125
Batch[905] - loss: 516.000000  dcg: 0.4927 mrr: 0.4833 auc: 0.2344 ^CExiting from training early
(pytorch) user10@430:~/lyx/src1$ python main.py -device 0 -test0 True -real-size=2
Using backend: pytorch
Loading data...
Parameters:
        BATCH_SIZE=32
        CANDIDATE_SIZE=10
        CATEGORY_NUM=500
        CLICK_SIZE=30
        CUDA=True
        DECAY=0.001
        DEVICE=0
        DROPOUT=0.5
        EARLY_STOPPING=1000
        EMBEDDING_DIM=256
        EMBEDDING_MID_DIM=48
        KERNEL_SIZE=3
        LOG_INTERVAL=1
        LR=0.0002
        MAX_NORM=3.0
        MULTICHANNEL=False
        NEWS_LIST_LENGTH=1025
        NODE_DROPOUT=0.1
        NON_STATIC=False
        NUM_ATTENTION_HEADS=16
        NUM_EPOCHS=10
        NUM_FILTERS=256
        NUM_WORDS_TITLE=100
        PRETRAINED_NAME=sgns.zhihu.word
        PRETRAINED_PATH=pretrained
        QUERY_VECTOR_DIM=128
        REAL_SIZE=2
        REFUSE_RATE=0.5
        REFUSE_SIZE=3
        SAVE_BEST=True
        SAVE_DIR=snapshot
        SNAPSHOT=None
        STATIC=False
        TEST0=True
        TEST_INTERVAL=100
        TESTZHIHU=False
        VOCAB_SIZE=250000
Traceback (most recent call last):
  File "main.py", line 142, in <module>
    train.train(batch_train, model, args, test,news_content,content,news_entity,entity)
  File "/home/user10/lyx/src1/train.py", line 30, in train
    possible,loss,news_vector,user_vector = model(news_adj, news_list, click, refuse, candidate, news_c_adj ,category_adj)
  File "/opt/miniconda3/envs/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/user10/lyx/src1/model0.py", line 85, in forward
    user_vector)
  File "/opt/miniconda3/envs/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/user10/lyx/src1/model0.py", line 386, in forward
    user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)/user_vector.shape()[-1]
TypeError: 'torch.Size' object is not callable
(pytorch) user10@430:~/lyx/src1$ python main.py -device 0 -test0 True -real-size=2
Using backend: pytorch
Loading data...
Parameters:
        BATCH_SIZE=32
        CANDIDATE_SIZE=10
        CATEGORY_NUM=500
        CLICK_SIZE=30
        CUDA=True
        DECAY=0.001
        DEVICE=0
        DROPOUT=0.5
        EARLY_STOPPING=1000
        EMBEDDING_DIM=256
        EMBEDDING_MID_DIM=48
        KERNEL_SIZE=3
        LOG_INTERVAL=1
        LR=0.0002
        MAX_NORM=3.0
        MULTICHANNEL=False
        NEWS_LIST_LENGTH=1025
        NODE_DROPOUT=0.1
        NON_STATIC=False
        NUM_ATTENTION_HEADS=16
        NUM_EPOCHS=10
        NUM_FILTERS=256
        NUM_WORDS_TITLE=100
        PRETRAINED_NAME=sgns.zhihu.word
        PRETRAINED_PATH=pretrained
        QUERY_VECTOR_DIM=128
        REAL_SIZE=2
        REFUSE_RATE=0.5
        REFUSE_SIZE=3
        SAVE_BEST=True
        SAVE_DIR=snapshot
        SNAPSHOT=None
        STATIC=False
        TEST0=True
        TEST_INTERVAL=100
        TESTZHIHU=False
        VOCAB_SIZE=250000
Batch[100] - loss: 490.450684  dcg: 0.6116 mrr: 0.5376 auc: 0.5488
Evaluation - loss: 499.810242  dcg: 59.8268 mrr: 49.5739 auc: 54.9899
Saving best model, acc: 59.8268%

Batch[200] - loss: 448.138367  dcg: 0.6416 mrr: 0.5718 auc: 0.5723 tensor([[[9.7490e-03, 8.0147e-03, 7.1274e-03,  ..., 4.5002e-03,
          3.9173e-03, 8.6620e+00],
         [8.3628e-03, 7.1917e-01, 3.8877e-01,  ..., 5.7439e-02,
          4.2620e-01, 7.6891e+00],
         [1.0643e-01, 2.0421e-01, 7.7155e-02,  ..., 1.0725e-01,
          2.7271e-02, 1.7852e+00],
         ...,
         [2.1743e-02, 7.2077e-01, 2.7184e-01,  ..., 1.1792e-01,
          3.8166e-01, 5.3494e+00],
         [1.3043e+00, 9.9393e-01, 9.2759e-01,  ..., 2.2933e-01,
          1.2461e+00, 2.2602e+00],
         [1.2811e-02, 3.3181e-01, 2.0534e-01,  ..., 1.8301e-01,
          4.0094e-01, 3.5285e+00]],

        [[2.6252e-02, 1.7147e-02, 1.0088e-03,  ..., 3.1199e-04,
          1.7154e+00, 1.7876e+00],
         [8.5206e-02, 1.0940e+00, 3.7249e-01,  ..., 4.8844e-01,
          3.0493e-01, 2.0283e+00],
         [9.9500e-04, 2.7365e-01, 5.0440e-03,  ..., 5.1323e-01,
          7.8903e-01, 2.1551e+00],
         ...,
         [5.4266e-02, 6.7313e-02, 1.3595e+00,  ..., 3.1438e-01,
          5.7680e-01, 4.1266e+00],
         [3.8936e-02, 2.5654e-02, 9.5655e-02,  ..., 1.7785e-01,
          1.3540e-01, 3.8414e+00],
         [1.2893e-03, 9.6939e-01, 5.6457e-01,  ..., 8.1195e-02,
          5.8279e-01, 4.4302e+00]],

        [[8.8546e-02, 1.3843e+00, 7.3625e-01,  ..., 9.3302e-02,
          4.0466e-01, 2.2674e+00],
         [1.6078e-03, 3.1138e-01, 1.0882e-01,  ..., 6.4684e-01,
          1.4133e-02, 3.9503e+00],
         [2.6652e-02, 6.3904e-01, 5.7894e-01,  ..., 1.5733e-01,
          1.2627e+00, 1.4284e+00],
         ...,
         [7.1985e-02, 6.1659e-01, 4.5633e-01,  ..., 2.4390e-01,
          4.0817e-01, 3.8401e+00],
         [4.9985e-02, 6.5635e-01, 3.4718e-01,  ..., 6.0870e-02,
          3.6653e-01, 4.2422e+00],
         [1.5961e-01, 4.9372e-02, 1.8450e-01,  ..., 5.7402e-02,
          9.0470e-02, 2.6971e+00]],

        ...,

        [[4.0852e-02, 2.8363e-01, 9.7942e-02,  ..., 4.3349e-06,
          6.5928e-04, 1.0831e+00],
         [8.5928e-03, 7.0765e-02, 6.9523e-02,  ..., 6.4769e-02,
          4.5717e-01, 1.5960e+00],
         [8.8512e-03, 6.4736e-01, 2.4366e-01,  ..., 9.1439e-02,
          8.3653e-02, 4.7559e+00],
         ...,
         [8.0273e-02, 4.3776e-01, 3.1397e-01,  ..., 1.8822e-01,
          1.5021e-01, 3.3978e+00],
         [1.7454e-04, 1.7678e-01, 1.1625e-01,  ..., 7.7935e-02,
          2.5949e-01, 1.4246e+00],
         [6.9961e-02, 1.0990e-01, 2.4662e-01,  ..., 1.8210e-01,
          3.0924e-01, 5.0275e+00]],

        [[8.5206e-02, 1.0940e+00, 3.7249e-01,  ..., 4.8844e-01,
          3.0493e-01, 2.0283e+00],
         [3.9122e-04, 2.0050e-03, 7.5768e-04,  ..., 1.3448e-03,
          2.2383e+00, 1.9984e+00],
         [1.4577e-01, 3.0448e-01, 2.2017e-01,  ..., 8.7800e-02,
          9.8537e-02, 3.8935e+00],
         ...,
         [1.9003e-03, 1.0446e+00, 1.0832e-01,  ..., 1.6302e-01,
          2.9820e-01, 3.7997e+00],
         [3.9698e-03, 4.7857e-01, 2.6527e-01,  ..., 1.4021e-01,
          4.0205e-01, 2.5572e+00],
         [1.4967e-03, 3.3067e-03, 7.0481e-03,  ..., 6.7811e-03,
          4.8264e-03, 1.5220e+00]],

        [[3.1794e-03, 2.0775e+00, 5.6037e-01,  ..., 9.8691e-02,
          8.6474e-01, 6.8982e+00],
         [4.3322e-02, 8.8551e-01, 7.6503e-01,  ..., 1.6611e-01,
          1.3524e-01, 3.0491e+00],
         [5.4849e-02, 6.2343e-01, 3.7128e-01,  ..., 5.1935e-01,
          5.8544e-01, 3.4502e+00],
         ...,
         [5.0790e-02, 6.4376e-02, 3.8945e-02,  ..., 4.7497e-03,
          7.5944e-02, 4.2001e+00],
         [9.2258e-04, 2.0983e+00, 6.3192e-01,  ..., 1.9073e-01,
          3.2683e-01, 3.9640e+00],
         [8.2813e-03, 1.0227e+00, 4.4256e-01,  ..., 3.0352e-02,
          1.0420e-01, 5.6441e+00]]], device='cuda:0', grad_fn=<CatBackward>) tensor([[9.7490e-03, 8.0147e-03, 7.1274e-03,  ..., 4.5002e-03, 3.9173e-03,
         8.6620e+00],
        [8.3628e-03, 7.1917e-01, 3.8877e-01,  ..., 5.7439e-02, 4.2620e-01,
         7.6891e+00],
        [1.0643e-01, 2.0421e-01, 7.7155e-02,  ..., 1.0725e-01, 2.7271e-02,
         1.7852e+00],
        ...,
        [2.1743e-02, 7.2077e-01, 2.7184e-01,  ..., 1.1792e-01, 3.8166e-01,
         5.3494e+00],
        [1.3043e+00, 9.9393e-01, 9.2759e-01,  ..., 2.2933e-01, 1.2461e+00,
         2.2602e+00],
        [1.2811e-02, 3.3181e-01, 2.0534e-01,  ..., 1.8301e-01, 4.0094e-01,
         3.5285e+00]], device='cuda:0', grad_fn=<SelectBackward>) tensor([9.7490e-03, 8.0147e-03, 7.1274e-03, 3.4154e-05, 1.4827e-02, 7.8705e+00,
        1.3874e+00, 5.3368e-04, 2.6726e+00, 1.3542e+00, 7.5738e-02, 1.3000e-05,
        3.9333e-03, 5.4458e-03, 5.6583e+00, 4.3187e-03, 5.4477e+00, 5.9373e-03,
        1.0269e+01, 2.1371e-03, 6.8181e+00, 1.0141e+01, 1.6866e-02, 6.4184e-03,
        9.0324e+00, 9.1491e-03, 4.0372e-03, 5.8199e-03, 5.4237e-03, 4.6681e+00,
        3.0578e+00, 7.8561e-03, 9.7096e-03, 1.7586e-02, 7.6351e-04, 4.2820e+00,
        2.9947e-03, 2.9803e-03, 9.7338e+00, 2.0883e-03, 6.7828e-03, 7.9650e+00,
        5.4370e-03, 4.8889e+00, 3.4563e+00, 3.6603e-04, 1.2286e+00, 6.6411e+00,
        1.2348e+00, 4.1015e+00, 5.2640e+00, 9.7983e+00, 5.0436e+00, 8.5136e+00,
        7.8911e-03, 8.8333e+00, 9.0255e+00, 9.1049e-03, 3.5178e-03, 1.0882e-02,
        2.6734e+00, 3.3090e-03, 8.5851e+00, 7.5299e+00, 4.3587e+00, 6.8971e+00,
        4.5016e-04, 2.2474e-02, 2.8749e+00, 2.6288e+00, 1.1168e+01, 1.7048e-08,
        6.7645e+00, 6.0897e+00, 5.0711e+00, 2.6671e+00, 1.3954e-02, 4.9449e-03,
        9.4687e-01, 9.2020e+00, 1.4739e-03, 1.9792e-03, 6.9975e+00, 8.5940e+00,
        5.9720e-03, 6.3737e+00, 1.0923e+00, 2.6878e+00, 1.0207e-02, 2.7549e+00,
        5.9901e+00, 1.5073e-01, 1.4580e-03, 4.7522e+00, 5.7163e+00, 7.9476e+00,
        7.7125e-03, 3.7109e-03, 1.8275e-03, 7.0733e-03, 6.7098e-03, 8.1821e+00,
        1.2300e+00, 2.6520e+00, 6.2065e-03, 4.5045e+00, 9.1209e+00, 3.1471e-03,
        4.4722e+00, 1.8230e+00, 5.7804e+00, 4.9989e+00, 2.8518e-04, 9.9115e+00,
        3.3307e-03, 3.7419e+00, 3.6676e+00, 7.4725e+00, 5.7702e+00, 5.7102e+00,
        2.7780e+00, 5.8852e+00, 7.4980e+00, 6.7813e-03, 7.5806e+00, 1.2226e+00,
        2.7562e+00, 9.9724e-03, 6.9960e-04, 1.0289e-03, 1.1988e+00, 7.1270e+00,
        8.2261e+00, 6.7718e+00, 1.4967e-03, 4.9689e+00, 1.5268e-02, 7.4574e+00,
        7.3016e+00, 4.6610e-03, 2.7976e-03, 8.1213e+00, 2.8069e-04, 4.2446e+00,
        2.7513e-05, 2.8906e+00, 4.7012e+00, 6.8636e+00, 1.1288e-02, 4.9939e-03,
        7.3760e+00, 5.7657e+00, 2.7817e-03, 3.4815e+00, 2.2181e+00, 1.0315e+01,
        8.0200e+00, 9.5622e+00, 9.4249e+00, 1.5855e+00, 4.0823e-01, 3.1844e+00,
        6.1004e+00, 5.9185e-01, 6.8250e+00, 1.8310e+00, 2.7212e-03, 2.6354e-03,
        1.6013e-03, 3.8381e+00, 1.7231e-04, 4.5053e+00, 7.9621e+00, 7.0223e+00,
        7.1341e-03, 1.6509e-02, 3.7532e-04, 1.0027e-03, 8.8561e+00, 2.8603e+00,
        9.2801e+00, 3.3670e+00, 1.7256e-04, 8.1792e-03, 8.8544e+00, 1.0208e-04,
        1.2279e-03, 8.7157e-03, 3.2372e-01, 7.1106e+00, 6.3804e+00, 2.5859e-04,
        1.0544e-05, 8.1065e+00, 1.3676e-02, 5.0164e-03, 7.1190e-03, 4.1003e+00,
        1.3629e+00, 2.1139e-02, 1.9928e+00, 1.0027e+01, 6.2241e+00, 5.3705e+00,
        1.0461e+01, 8.4262e-03, 4.1526e-04, 2.2351e+00, 1.9730e+00, 3.1036e-04,
        6.1445e+00, 7.9000e-03, 9.6933e-03, 1.2197e-03, 3.3022e-03, 5.5120e-03,
        2.0212e-04, 3.5541e-03, 3.2390e-04, 1.4759e-02, 4.7422e+00, 9.5450e-01,
        4.2404e-03, 1.0813e-02, 1.5305e-02, 8.8432e-01, 3.3514e+00, 6.6029e+00,
        3.2253e+00, 1.3049e+00, 1.8241e+00, 6.6907e+00, 3.4002e+00, 1.9502e-02,
        7.6794e+00, 4.9376e-03, 7.4514e-03, 5.5694e+00, 2.2575e+00, 5.8857e-03,
        2.4300e-01, 4.2663e-01, 4.6938e+00, 6.6957e-03, 9.6401e-01, 2.4543e-03,
        1.3376e+00, 8.3546e+00, 4.5533e+00, 8.7142e-03, 3.9586e-04, 5.9792e+00,
        7.2540e+00, 4.5002e-03, 3.9173e-03, 8.6620e+00], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([[0.0235, 0.9502, 0.3657,  ..., 0.1271, 0.1882, 3.9557],
        [0.0586, 0.8869, 0.5113,  ..., 0.1476, 0.5325, 5.0708],
        [0.0531, 0.5944, 0.2159,  ..., 0.0694, 0.3015, 4.0663],
        ...,
        [0.2209, 0.3084, 0.2488,  ..., 0.2301, 0.4584, 3.5567],
        [0.0279, 1.0983, 0.4594,  ..., 0.1829, 0.2920, 5.1328],
        [0.1637, 0.5785, 0.3049,  ..., 0.1640, 0.2786, 3.7992]],
       device='cuda:0', grad_fn=<SqueezeBackward1>) tensor([0.0235, 0.9502, 0.3657, 0.2656, 0.2486, 3.1667, 0.7417, 0.2922, 0.4455,
        0.2671, 0.1876, 0.0348, 0.2092, 0.1448, 3.2064, 0.0310, 2.9991, 0.0326,
        3.9548, 0.1040, 3.8882, 4.5235, 0.0949, 0.0443, 4.0083, 0.1116, 0.0738,
        0.4102, 0.6107, 3.3801, 0.3611, 0.3887, 0.3338, 2.3845, 0.4315, 3.2369,
        0.6543, 0.9541, 4.2048, 0.1747, 0.1114, 4.0959, 0.1488, 3.4330, 3.2035,
        0.0298, 0.9452, 3.9778, 0.4337, 2.5173, 3.8436, 4.6169, 3.2125, 3.9544,
        0.1319, 3.3331, 4.3062, 0.0460, 0.0674, 0.2533, 0.3780, 0.2333, 3.8989,
        3.6434, 2.9964, 4.1974, 0.1756, 0.0888, 2.7284, 0.5541, 3.7704, 0.8678,
        3.8945, 3.4834, 0.6782, 0.6070, 0.7586, 0.4123, 1.2848, 4.0944, 0.2289,
        0.1589, 3.1752, 3.2311, 0.6293, 4.0654, 2.2646, 3.0071, 0.0637, 3.0872,
        3.1170, 2.3850, 0.0369, 3.7836, 3.7978, 3.2488, 0.0625, 0.0744, 0.0282,
        1.3196, 0.0054, 4.0027, 0.7310, 1.9704, 0.1005, 3.2659, 4.5110, 0.8660,
        3.4244, 3.3867, 4.0975, 1.4621, 0.0161, 4.2451, 0.0446, 2.4673, 0.7541,
        3.8832, 4.3989, 2.6733, 0.4820, 0.7604, 4.0556, 1.7044, 4.4337, 1.8023,
        3.4089, 0.0622, 1.2602, 0.4455, 0.3090, 3.2751, 3.6915, 4.3786, 0.2057,
        3.1013, 0.1299, 3.3246, 4.8028, 0.0747, 0.2033, 4.1173, 0.0639, 3.2107,
        0.2097, 0.4728, 3.3812, 3.2920, 0.2535, 0.0852, 4.0809, 4.5874, 0.5197,
        0.6211, 2.8753, 4.4784, 4.1633, 3.4341, 4.3222, 2.7586, 0.1346, 3.2442,
        3.9028, 0.3310, 3.7956, 0.3628, 0.1871, 0.0490, 0.1915, 2.5708, 0.8917,
        3.9430, 3.7191, 3.8267, 0.2848, 0.8838, 0.0544, 0.3216, 3.4106, 3.5769,
        4.0974, 0.7613, 0.0137, 0.0496, 3.9356, 0.0508, 0.1384, 1.5850, 0.3642,
        4.7019, 3.9529, 0.0572, 0.1129, 3.6793, 0.2804, 0.0794, 0.1975, 3.3831,
        0.5574, 0.2931, 1.0633, 4.5843, 3.6607, 0.9613, 4.4363, 0.0256, 0.0367,
        0.2405, 2.6051, 0.3659, 3.7071, 0.9772, 0.3004, 0.0604, 1.2280, 0.0541,
        0.0526, 1.0236, 0.0374, 0.1152, 0.5512, 0.6335, 0.2366, 0.0796, 0.0999,
        0.2558, 2.8682, 2.4071, 3.2613, 0.4830, 1.0275, 4.4169, 3.6800, 0.3796,
        3.7042, 0.1265, 0.0592, 2.6956, 2.5378, 1.4960, 2.3221, 0.1305, 2.7008,
        0.0472, 3.2843, 0.1574, 0.3645, 4.4463, 3.3822, 0.2746, 0.0888, 3.8305,
        3.8116, 0.1271, 0.1882, 3.9557], device='cuda:0',
       grad_fn=<SelectBackward>)
mrr tensor([1.6467e-03, 1.4674e-01, 2.2098e-03, 8.2854e-04, 2.0497e-02, 6.6287e-01,
        4.2256e-04, 6.8752e-04, 5.1387e-04, 1.6358e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.1467, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.6467e-03, 1.4674e-01, 2.2098e-03, 8.2854e-04, 2.0497e-02, 6.6287e-01,
        4.2256e-04, 6.8752e-04, 5.1387e-04, 1.6358e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 9, 1, 4, 2, 0, 3, 7, 8, 6) tensor([0.8562])
auc tensor([1.6467e-03, 1.4674e-01, 2.2098e-03, 8.2854e-04, 2.0497e-02, 6.6287e-01,
        4.2256e-04, 6.8752e-04, 5.1387e-04, 1.6358e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([1.2958e-06, 9.9976e-01, 1.5050e-04, 6.0177e-07, 2.3105e-05, 4.8044e-05,
        4.5110e-06, 4.6051e-06, 2.4267e-06, 5.2459e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9998, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.2958e-06, 9.9976e-01, 1.5050e-04, 6.0177e-07, 2.3105e-05, 4.8044e-05,
        4.5110e-06, 4.6051e-06, 2.4267e-06, 5.2459e-07], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 2, 5, 4, 7, 6, 8, 0, 3, 9) tensor([1.3155])
auc tensor([1.2958e-06, 9.9976e-01, 1.5050e-04, 6.0177e-07, 2.3105e-05, 4.8044e-05,
        4.5110e-06, 4.6051e-06, 2.4267e-06, 5.2459e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([2.2952e-03, 3.3692e-01, 8.9701e-04, 3.8569e-01, 5.5974e-04, 5.6767e-02,
        5.4217e-02, 2.2198e-04, 4.1605e-03, 1.5827e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.3369, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.2952e-03, 3.3692e-01, 8.9701e-04, 3.8569e-01, 5.5974e-04, 5.6767e-02,
        5.4217e-02, 2.2198e-04, 4.1605e-03, 1.5827e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 1, 9, 5, 6, 8, 0, 2, 4, 7) tensor([0.9643])
auc tensor([2.2952e-03, 3.3692e-01, 8.9701e-04, 3.8569e-01, 5.5974e-04, 5.6767e-02,
        5.4217e-02, 2.2198e-04, 4.1605e-03, 1.5827e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([9.8290e-01, 1.6970e-02, 2.5473e-06, 5.2395e-05, 7.7491e-06, 1.2457e-07,
        1.3641e-05, 1.8669e-06, 9.2817e-06, 3.7829e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9829, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.8290e-01, 1.6970e-02, 2.5473e-06, 5.2395e-05, 7.7491e-06, 1.2457e-07,
        1.3641e-05, 1.8669e-06, 9.2817e-06, 3.7829e-05], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 3, 9, 6, 8, 4, 2, 7, 5) tensor([1.6309])
auc tensor([9.8290e-01, 1.6970e-02, 2.5473e-06, 5.2395e-05, 7.7491e-06, 1.2457e-07,
        1.3641e-05, 1.8669e-06, 9.2817e-06, 3.7829e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([6.9900e-03, 8.5300e-01, 4.2394e-03, 4.9385e-03, 1.6017e-02, 1.4243e-02,
        3.2301e-02, 6.8048e-02, 1.1703e-04, 1.0367e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.8530, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([6.9900e-03, 8.5300e-01, 4.2394e-03, 4.9385e-03, 1.6017e-02, 1.4243e-02,
        3.2301e-02, 6.8048e-02, 1.1703e-04, 1.0367e-04], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 7, 6, 4, 5, 0, 3, 2, 8, 9) tensor([1.3562])
auc tensor([6.9900e-03, 8.5300e-01, 4.2394e-03, 4.9385e-03, 1.6017e-02, 1.4243e-02,
        3.2301e-02, 6.8048e-02, 1.1703e-04, 1.0367e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([9.7726e-01, 1.2287e-02, 7.5403e-03, 1.4522e-05, 1.4576e-04, 6.9005e-04,
        4.8227e-04, 2.0338e-04, 4.8449e-04, 8.9311e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9773, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.7726e-01, 1.2287e-02, 7.5403e-03, 1.4522e-05, 1.4576e-04, 6.9005e-04,
        4.8227e-04, 2.0338e-04, 4.8449e-04, 8.9311e-04], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 2, 9, 5, 8, 6, 7, 4, 3) tensor([1.6309])
auc tensor([9.7726e-01, 1.2287e-02, 7.5403e-03, 1.4522e-05, 1.4576e-04, 6.9005e-04,
        4.8227e-04, 2.0338e-04, 4.8449e-04, 8.9311e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([0.0101, 0.0543, 0.0008, 0.0030, 0.0026, 0.6437, 0.0011, 0.0007, 0.0007,
        0.2830], device='cuda:0', grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.0543, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0101, 0.0543, 0.0008, 0.0030, 0.0026, 0.6437, 0.0011, 0.0007, 0.0007,
        0.2830], device='cuda:0', grad_fn=<SelectBackward>) (5, 9, 1, 0, 3, 4, 6, 2, 7, 8) tensor([0.9307])
auc tensor([0.0101, 0.0543, 0.0008, 0.0030, 0.0026, 0.6437, 0.0011, 0.0007, 0.0007,
        0.2830], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([3.2850e-02, 1.3282e-02, 7.7288e-01, 1.0406e-02, 4.1849e-04, 5.7780e-03,
        2.2738e-04, 3.7268e-04, 8.8012e-04, 1.6291e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.0328, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.2850e-02, 1.3282e-02, 7.7288e-01, 1.0406e-02, 4.1849e-04, 5.7780e-03,
        2.2738e-04, 3.7268e-04, 8.8012e-04, 1.6291e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 9, 0, 1, 3, 5, 8, 4, 7, 6) tensor([0.9307])
auc tensor([3.2850e-02, 1.3282e-02, 7.7288e-01, 1.0406e-02, 4.1849e-04, 5.7780e-03,
        2.2738e-04, 3.7268e-04, 8.8012e-04, 1.6291e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([2.0197e-02, 9.7811e-01, 1.3335e-07, 1.5634e-06, 2.0109e-06, 3.0001e-05,
        1.6482e-03, 2.2200e-07, 8.9476e-09, 1.0918e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9781, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.0197e-02, 9.7811e-01, 1.3335e-07, 1.5634e-06, 2.0109e-06, 3.0001e-05,
        1.6482e-03, 2.2200e-07, 8.9476e-09, 1.0918e-05], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 6, 5, 9, 4, 3, 7, 2, 8) tensor([1.6309])
auc tensor([2.0197e-02, 9.7811e-01, 1.3335e-07, 1.5634e-06, 2.0109e-06, 3.0001e-05,
        1.6482e-03, 2.2200e-07, 8.9476e-09, 1.0918e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([1.4478e-03, 8.1919e-04, 1.0396e-03, 8.8405e-03, 9.3045e-01, 1.5468e-03,
        4.8943e-02, 3.3512e-04, 2.5430e-03, 4.0358e-03], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([7], dtype=torch.int32) tensor(0.0014, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.4478e-03, 8.1919e-04, 1.0396e-03, 8.8405e-03, 9.3045e-01, 1.5468e-03,
        4.8943e-02, 3.3512e-04, 2.5430e-03, 4.0358e-03], device='cuda:0',
       grad_fn=<SelectBackward>) (4, 6, 3, 9, 8, 5, 0, 2, 1, 7) tensor([0.6344])
auc tensor([1.4478e-03, 8.1919e-04, 1.0396e-03, 8.8405e-03, 9.3045e-01, 1.5468e-03,
        4.8943e-02, 3.3512e-04, 2.5430e-03, 4.0358e-03], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.1875])
mrr tensor([3.3618e-04, 1.0651e-03, 3.3015e-05, 8.0817e-01, 2.1143e-04, 3.1505e-05,
        2.0531e-04, 1.4958e-02, 1.7492e-01, 7.0717e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0.0011, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.3618e-04, 1.0651e-03, 3.3015e-05, 8.0817e-01, 2.1143e-04, 3.1505e-05,
        2.0531e-04, 1.4958e-02, 1.7492e-01, 7.0717e-05], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 8, 7, 1, 0, 4, 6, 9, 2, 5) tensor([0.8175])
auc tensor([3.3618e-04, 1.0651e-03, 3.3015e-05, 8.0817e-01, 2.1143e-04, 3.1505e-05,
        2.0531e-04, 1.4958e-02, 1.7492e-01, 7.0717e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([1.1534e-06, 6.8902e-09, 8.7138e-09, 4.8070e-08, 9.1890e-10, 1.4600e-07,
        5.0391e-01, 4.9609e-01, 1.0839e-07, 2.3164e-09], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(1.1534e-06, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.1534e-06, 6.8902e-09, 8.7138e-09, 4.8070e-08, 9.1890e-10, 1.4600e-07,
        5.0391e-01, 4.9609e-01, 1.0839e-07, 2.3164e-09], device='cuda:0',
       grad_fn=<SelectBackward>) (6, 7, 0, 5, 8, 3, 2, 1, 9, 4) tensor([0.8155])
auc tensor([1.1534e-06, 6.8902e-09, 8.7138e-09, 4.8070e-08, 9.1890e-10, 1.4600e-07,
        5.0391e-01, 4.9609e-01, 1.0839e-07, 2.3164e-09], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0088, 0.1332, 0.1861, 0.0031, 0.0111, 0.0043, 0.0200, 0.0546, 0.5775,
        0.0012], device='cuda:0', grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.1332, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0088, 0.1332, 0.1861, 0.0031, 0.0111, 0.0043, 0.0200, 0.0546, 0.5775,
        0.0012], device='cuda:0', grad_fn=<SelectBackward>) (8, 2, 1, 7, 6, 4, 0, 5, 3, 9) tensor([0.8333])
auc tensor([0.0088, 0.1332, 0.1861, 0.0031, 0.0111, 0.0043, 0.0200, 0.0546, 0.5775,
        0.0012], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.5625])
mrr tensor([1.6061e-04, 3.6805e-05, 3.8204e-02, 1.6115e-04, 4.6974e-02, 8.9911e-01,
        3.5160e-05, 9.3541e-05, 4.4176e-05, 1.5179e-02], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([6], dtype=torch.int32) tensor(0.0002, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.6061e-04, 3.6805e-05, 3.8204e-02, 1.6115e-04, 4.6974e-02, 8.9911e-01,
        3.5160e-05, 9.3541e-05, 4.4176e-05, 1.5179e-02], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 4, 2, 9, 3, 0, 7, 8, 1, 6) tensor([0.6572])
auc tensor([1.6061e-04, 3.6805e-05, 3.8204e-02, 1.6115e-04, 4.6974e-02, 8.9911e-01,
        3.5160e-05, 9.3541e-05, 4.4176e-05, 1.5179e-02], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.2500])
mrr tensor([0.0845, 0.0105, 0.5978, 0.0147, 0.0097, 0.0102, 0.1260, 0.0222, 0.0055,
        0.1189], device='cuda:0', grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(0.0845, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0845, 0.0105, 0.5978, 0.0147, 0.0097, 0.0102, 0.1260, 0.0222, 0.0055,
        0.1189], device='cuda:0', grad_fn=<SelectBackward>) (2, 6, 9, 0, 7, 3, 1, 5, 4, 8) tensor([0.7640])
auc tensor([0.0845, 0.0105, 0.5978, 0.0147, 0.0097, 0.0102, 0.1260, 0.0222, 0.0055,
        0.1189], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([4.4640e-08, 6.1494e-07, 2.2309e-06, 1.3860e-01, 1.3644e-02, 6.0205e-07,
        7.2850e-01, 1.1925e-01, 5.7608e-09, 2.2322e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([7], dtype=torch.int32) tensor(6.1494e-07, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([4.4640e-08, 6.1494e-07, 2.2309e-06, 1.3860e-01, 1.3644e-02, 6.0205e-07,
        7.2850e-01, 1.1925e-01, 5.7608e-09, 2.2322e-06], device='cuda:0',
       grad_fn=<SelectBackward>) (6, 3, 7, 4, 9, 2, 1, 5, 0, 8) tensor([0.6344])
auc tensor([4.4640e-08, 6.1494e-07, 2.2309e-06, 1.3860e-01, 1.3644e-02, 6.0205e-07,
        7.2850e-01, 1.1925e-01, 5.7608e-09, 2.2322e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.1875])
mrr tensor([1.3418e-01, 1.7253e-02, 3.6148e-03, 8.1552e-01, 9.5724e-05, 6.0967e-04,
        1.8355e-04, 3.8441e-05, 1.1214e-02, 1.7291e-02], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.1342, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.3418e-01, 1.7253e-02, 3.6148e-03, 8.1552e-01, 9.5724e-05, 6.0967e-04,
        1.8355e-04, 3.8441e-05, 1.1214e-02, 1.7291e-02], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 0, 9, 1, 8, 2, 5, 6, 4, 7) tensor([1.0616])
auc tensor([1.3418e-01, 1.7253e-02, 3.6148e-03, 8.1552e-01, 9.5724e-05, 6.0967e-04,
        1.8355e-04, 3.8441e-05, 1.1214e-02, 1.7291e-02], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8125])
mrr tensor([8.4404e-02, 6.6877e-03, 2.0961e-03, 1.7267e-03, 2.6571e-04, 3.0898e-01,
        3.9829e-04, 2.0995e-03, 5.9296e-01, 3.8972e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.0844, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([8.4404e-02, 6.6877e-03, 2.0961e-03, 1.7267e-03, 2.6571e-04, 3.0898e-01,
        3.9829e-04, 2.0995e-03, 5.9296e-01, 3.8972e-04], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 5, 0, 1, 7, 2, 3, 6, 9, 4) tensor([0.9307])
auc tensor([8.4404e-02, 6.6877e-03, 2.0961e-03, 1.7267e-03, 2.6571e-04, 3.0898e-01,
        3.9829e-04, 2.0995e-03, 5.9296e-01, 3.8972e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([8.2102e-02, 3.0620e-05, 3.5745e-04, 1.2304e-04, 8.3774e-04, 7.0452e-05,
        1.3510e-03, 1.1132e-03, 5.0692e-05, 9.1396e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.0821, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([8.2102e-02, 3.0620e-05, 3.5745e-04, 1.2304e-04, 8.3774e-04, 7.0452e-05,
        1.3510e-03, 1.1132e-03, 5.0692e-05, 9.1396e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (9, 0, 6, 7, 4, 2, 3, 5, 8, 1) tensor([0.9200])
auc tensor([8.2102e-02, 3.0620e-05, 3.5745e-04, 1.2304e-04, 8.3774e-04, 7.0452e-05,
        1.3510e-03, 1.1132e-03, 5.0692e-05, 9.1396e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([0.0095, 0.0195, 0.0861, 0.0509, 0.0023, 0.0089, 0.7195, 0.0413, 0.0551,
        0.0068], device='cuda:0', grad_fn=<SelectBackward>) tensor([6], dtype=torch.int32) tensor(0.0195, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0095, 0.0195, 0.0861, 0.0509, 0.0023, 0.0089, 0.7195, 0.0413, 0.0551,
        0.0068], device='cuda:0', grad_fn=<SelectBackward>) (6, 2, 8, 3, 7, 1, 0, 5, 9, 4) tensor([0.6895])
auc tensor([0.0095, 0.0195, 0.0861, 0.0509, 0.0023, 0.0089, 0.7195, 0.0413, 0.0551,
        0.0068], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([3.4985e-03, 2.0085e-05, 1.7776e-02, 2.1444e-03, 2.2759e-05, 6.8872e-01,
        1.4718e-02, 4.5989e-05, 1.2197e-03, 2.7183e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(0.0035, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.4985e-03, 2.0085e-05, 1.7776e-02, 2.1444e-03, 2.2759e-05, 6.8872e-01,
        1.4718e-02, 4.5989e-05, 1.2197e-03, 2.7183e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 9, 2, 6, 0, 3, 8, 7, 4, 1) tensor([0.6759])
auc tensor([3.4985e-03, 2.0085e-05, 1.7776e-02, 2.1444e-03, 2.2759e-05, 6.8872e-01,
        1.4718e-02, 4.5989e-05, 1.2197e-03, 2.7183e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.2500])
mrr tensor([5.9768e-04, 1.3651e-03, 6.6264e-06, 4.0196e-07, 6.9168e-06, 2.1935e-06,
        4.5418e-04, 4.0311e-04, 9.9662e-01, 5.4763e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([2], dtype=torch.int32) tensor(0.0014, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.9768e-04, 1.3651e-03, 6.6264e-06, 4.0196e-07, 6.9168e-06, 2.1935e-06,
        4.5418e-04, 4.0311e-04, 9.9662e-01, 5.4763e-04], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 1, 0, 9, 6, 7, 4, 2, 5, 3) tensor([1.1309])
auc tensor([5.9768e-04, 1.3651e-03, 6.6264e-06, 4.0196e-07, 6.9168e-06, 2.1935e-06,
        4.5418e-04, 4.0311e-04, 9.9662e-01, 5.4763e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8750])
mrr tensor([5.6499e-01, 2.2970e-01, 1.7659e-01, 3.3951e-03, 2.4399e-04, 7.9279e-03,
        4.7377e-03, 1.1372e-02, 8.8549e-04, 1.6220e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.5650, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.6499e-01, 2.2970e-01, 1.7659e-01, 3.3951e-03, 2.4399e-04, 7.9279e-03,
        4.7377e-03, 1.1372e-02, 8.8549e-04, 1.6220e-04], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 2, 7, 5, 6, 3, 8, 4, 9) tensor([1.6309])
auc tensor([5.6499e-01, 2.2970e-01, 1.7659e-01, 3.3951e-03, 2.4399e-04, 7.9279e-03,
        4.7377e-03, 1.1372e-02, 8.8549e-04, 1.6220e-04], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([4.2649e-07, 9.8793e-01, 9.8962e-05, 6.8576e-05, 4.0421e-03, 5.2600e-04,
        6.6960e-03, 5.9289e-05, 5.7745e-04, 2.1795e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9879, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([4.2649e-07, 9.8793e-01, 9.8962e-05, 6.8576e-05, 4.0421e-03, 5.2600e-04,
        6.6960e-03, 5.9289e-05, 5.7745e-04, 2.1795e-06], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 4, 8, 5, 2, 3, 7, 9, 0) tensor([1.2891])
auc tensor([4.2649e-07, 9.8793e-01, 9.8962e-05, 6.8576e-05, 4.0421e-03, 5.2600e-04,
        6.6960e-03, 5.9289e-05, 5.7745e-04, 2.1795e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([3.4299e-03, 4.8070e-02, 1.9561e-03, 1.1050e-03, 2.1085e-03, 1.7847e-01,
        2.9327e-04, 1.7458e-01, 4.7189e-01, 1.1810e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(0.0481, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.4299e-03, 4.8070e-02, 1.9561e-03, 1.1050e-03, 2.1085e-03, 1.7847e-01,
        2.9327e-04, 1.7458e-01, 4.7189e-01, 1.1810e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 5, 7, 9, 1, 0, 4, 2, 3, 6) tensor([0.7431])
auc tensor([3.4299e-03, 4.8070e-02, 1.9561e-03, 1.1050e-03, 2.1085e-03, 1.7847e-01,
        2.9327e-04, 1.7458e-01, 4.7189e-01, 1.1810e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([9.9355e-01, 5.8451e-03, 3.3407e-08, 2.5291e-08, 9.9243e-05, 4.2185e-04,
        6.3907e-05, 1.7712e-05, 8.6014e-08, 9.5037e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9936, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.9355e-01, 5.8451e-03, 3.3407e-08, 2.5291e-08, 9.9243e-05, 4.2185e-04,
        6.3907e-05, 1.7712e-05, 8.6014e-08, 9.5037e-07], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 5, 4, 6, 7, 9, 8, 2, 3) tensor([1.6309])
auc tensor([9.9355e-01, 5.8451e-03, 3.3407e-08, 2.5291e-08, 9.9243e-05, 4.2185e-04,
        6.3907e-05, 1.7712e-05, 8.6014e-08, 9.5037e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([0.0037, 0.0071, 0.1344, 0.0174, 0.2946, 0.0072, 0.0031, 0.0065, 0.4172,
        0.1087], device='cuda:0', grad_fn=<SelectBackward>) tensor([7], dtype=torch.int32) tensor(0.0071, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0037, 0.0071, 0.1344, 0.0174, 0.2946, 0.0072, 0.0031, 0.0065, 0.4172,
        0.1087], device='cuda:0', grad_fn=<SelectBackward>) (8, 4, 2, 9, 3, 5, 1, 7, 0, 6) tensor([0.6344])
auc tensor([0.0037, 0.0071, 0.1344, 0.0174, 0.2946, 0.0072, 0.0031, 0.0065, 0.4172,
        0.1087], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.1875])
mrr tensor([1.0000e+00, 1.0558e-06, 6.1179e-07, 3.4394e-10, 1.7204e-09, 2.0480e-11,
        1.2584e-08, 3.5855e-09, 8.0717e-10, 1.3192e-08], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1.0000, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 1.0558e-06, 6.1179e-07, 3.4394e-10, 1.7204e-09, 2.0480e-11,
        1.2584e-08, 3.5855e-09, 8.0717e-10, 1.3192e-08], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 2, 9, 6, 7, 4, 8, 3, 5) tensor([1.6309])
auc tensor([1.0000e+00, 1.0558e-06, 6.1179e-07, 3.4394e-10, 1.7204e-09, 2.0480e-11,
        1.2584e-08, 3.5855e-09, 8.0717e-10, 1.3192e-08], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([0.0283, 0.0222, 0.1626, 0.0151, 0.0011, 0.2183, 0.0026, 0.3879, 0.0028,
        0.1592], device='cuda:0', grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(0.0283, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0283, 0.0222, 0.1626, 0.0151, 0.0011, 0.2183, 0.0026, 0.3879, 0.0028,
        0.1592], device='cuda:0', grad_fn=<SelectBackward>) (7, 5, 2, 9, 0, 1, 3, 8, 6, 4) tensor([0.7431])
auc tensor([0.0283, 0.0222, 0.1626, 0.0151, 0.0011, 0.2183, 0.0026, 0.3879, 0.0028,
        0.1592], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([0.0631, 0.1093, 0.0077, 0.0015, 0.0824, 0.4132, 0.0339, 0.0075, 0.2790,
        0.0024], device='cuda:0', grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.1093, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([0.0631, 0.1093, 0.0077, 0.0015, 0.0824, 0.4132, 0.0339, 0.0075, 0.2790,
        0.0024], device='cuda:0', grad_fn=<SelectBackward>) (5, 8, 1, 4, 0, 6, 2, 7, 9, 3) tensor([0.8869])
auc tensor([0.0631, 0.1093, 0.0077, 0.0015, 0.0824, 0.4132, 0.0339, 0.0075, 0.2790,
        0.0024], device='cuda:0', grad_fn=<SelectBackward>) tensor([0.6875])
mrr tensor([3.3542e-05, 2.5557e-05, 9.7779e-04, 4.1756e-04, 4.8325e-03, 6.7007e-02,
        1.5155e-01, 1.6195e-04, 1.0421e-03, 7.7395e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([9], dtype=torch.int32) tensor(3.3542e-05, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.3542e-05, 2.5557e-05, 9.7779e-04, 4.1756e-04, 4.8325e-03, 6.7007e-02,
        1.5155e-01, 1.6195e-04, 1.0421e-03, 7.7395e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (9, 6, 5, 4, 8, 2, 3, 7, 0, 1) tensor([0.5901])
auc tensor([3.3542e-05, 2.5557e-05, 9.7779e-04, 4.1756e-04, 4.8325e-03, 6.7007e-02,
        1.5155e-01, 1.6195e-04, 1.0421e-03, 7.7395e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])

Evaluation - loss: 464.043762  dcg: 63.2389 mrr: 55.7238 auc: 57.6046
Saving best model, acc: 63.2389%

Batch[300] - loss: 448.578979  dcg: 0.6533 mrr: 0.5779 auc: 0.6289
Evaluation - loss: 447.760193  dcg: 64.4633 mrr: 58.0782 auc: 58.3984
Saving best model, acc: 64.4633%

Batch[400] - loss: 404.842834  dcg: 0.6876 mrr: 0.6953 auc: 0.6016
Evaluation - loss: 442.325500  dcg: 65.4871 mrr: 59.6782 auc: 59.4506
Saving best model, acc: 65.4871%

Batch[500] - loss: 430.890198  dcg: 0.6659 mrr: 0.6298 auc: 0.5977
Evaluation - loss: 437.495300  dcg: 66.2254 mrr: 60.9755 auc: 60.2319
Saving best model, acc: 66.2254%

Batch[600] - loss: 448.367310  dcg: 0.6528 mrr: 0.5762 auc: 0.5957 tensor([[[7.1546e-02, 2.5499e-01, 2.1275e+00,  ..., 2.3257e-01,
          3.7934e-01, 1.1552e+01],
         [7.5823e-01, 7.4617e-05, 1.2352e-05,  ..., 1.0687e+00,
          2.9321e-06, 1.6589e+01],
         [5.1661e-01, 6.4545e-01, 4.6350e-01,  ..., 7.0831e-01,
          4.4300e-02, 7.3653e+00],
         ...,
         [3.4212e-01, 1.0389e+00, 3.9354e-01,  ..., 1.3082e+00,
          1.3878e+00, 6.5129e+00],
         [2.9431e-01, 1.0359e+00, 2.3197e+00,  ..., 8.1471e-01,
          8.5662e-01, 1.1298e+01],
         [9.9838e-03, 3.0187e-02, 9.4567e-03,  ..., 3.4208e-01,
          3.7252e-02, 7.4951e+00]],

        [[1.4275e-03, 3.1778e-03, 2.1604e-05,  ..., 8.1692e-08,
          1.4814e-01, 1.6845e+01],
         [6.4188e-01, 2.1174e-01, 1.1363e+00,  ..., 1.6551e-01,
          3.0030e-01, 1.0033e+01],
         [1.4476e+00, 7.7726e-01, 3.7793e-01,  ..., 1.1265e+00,
          7.0966e-02, 6.7111e+00],
         ...,
         [4.6253e-01, 1.0070e-01, 2.6825e-01,  ..., 1.1222e-01,
          3.8824e-01, 8.3948e+00],
         [3.1181e-01, 5.0326e-01, 6.1354e-01,  ..., 1.2501e+00,
          5.8183e-01, 4.6143e+00],
         [7.8154e-02, 1.1715e-02, 3.0154e-01,  ..., 6.0649e-01,
          1.1961e+00, 7.6994e+00]],

        [[7.2940e-02, 3.7722e+00, 2.3705e-01,  ..., 1.2547e+00,
          3.8132e-01, 4.8584e+00],
         [7.6672e-02, 2.2975e-01, 5.8500e-02,  ..., 3.3330e-01,
          1.9463e-01, 1.0983e+01],
         [2.8565e-02, 4.3774e-01, 1.8938e+00,  ..., 4.3430e-01,
          3.8725e-01, 8.7980e+00],
         ...,
         [6.7391e-01, 5.4818e-04, 3.3438e-03,  ..., 6.0048e-04,
          8.2737e-02, 8.1057e+00],
         [1.7696e-01, 1.1711e+00, 1.8943e+00,  ..., 4.9396e-01,
          8.9703e-01, 5.8821e+00],
         [1.4322e-01, 7.4876e-01, 1.7321e-01,  ..., 6.9331e-01,
          1.2781e-01, 1.0145e+01]],

        ...,

        [[1.8427e-01, 1.6548e+00, 1.1846e+00,  ..., 1.3787e+00,
          7.5768e-01, 5.0828e+00],
         [1.0293e+00, 4.7742e-02, 6.0721e-01,  ..., 5.4429e-01,
          4.0471e-01, 7.9524e+00],
         [2.6601e-01, 4.4622e-01, 8.6924e-01,  ..., 3.7245e-01,
          6.7874e-01, 9.9441e+00],
         ...,
         [4.1806e-02, 1.9178e-01, 3.4796e-01,  ..., 8.9364e-02,
          9.4127e-02, 9.6681e+00],
         [6.7634e-02, 6.9254e-01, 1.6242e+00,  ..., 1.6240e-04,
          5.2478e-01, 1.0147e+01],
         [1.6067e-01, 1.1287e+00, 1.3154e-01,  ..., 3.7243e+00,
          3.3406e-01, 6.2811e+00]],

        [[2.2710e-06, 1.2658e-01, 8.1033e-04,  ..., 1.7482e-01,
          7.5966e-01, 9.4277e+00],
         [5.5885e-01, 1.2316e-01, 5.4152e-01,  ..., 1.4855e-01,
          7.2784e-01, 2.0492e+01],
         [3.2559e-01, 2.8278e-01, 6.6816e-01,  ..., 3.6549e-01,
          1.3421e+00, 6.3179e+00],
         ...,
         [3.3130e-02, 9.3140e-01, 2.8548e-01,  ..., 5.1974e-01,
          3.5938e-02, 7.0685e+00],
         [1.2329e+00, 4.4237e-01, 1.0637e+00,  ..., 2.6802e-01,
          6.2810e-01, 1.1452e+01],
         [1.3618e-08, 3.3586e-01, 1.4495e-01,  ..., 2.8228e+00,
          3.7463e-01, 1.4499e+01]],

        [[3.1419e-05, 5.7916e-01, 5.0421e-05,  ..., 5.2142e+00,
          6.6020e-01, 2.4309e+00],
         [2.2914e-01, 8.8342e-03, 1.2639e+00,  ..., 3.1616e-01,
          1.7172e-01, 7.4282e+00],
         [6.9191e-01, 2.4549e-01, 1.0880e+00,  ..., 1.5116e-01,
          7.7427e-01, 3.8181e+00],
         ...,
         [6.8326e-01, 7.6429e-01, 6.6527e-01,  ..., 2.2548e-01,
          1.5257e-01, 7.6999e+00],
         [1.4086e-01, 4.9414e-02, 2.0928e+00,  ..., 2.8207e-01,
          2.3129e-01, 2.1659e+01],
         [6.4864e-04, 1.4430e-03, 2.1676e-02,  ..., 5.1358e+00,
          1.6486e+00, 5.2946e+00]]], device='cuda:0', grad_fn=<CatBackward>) tensor([[7.1546e-02, 2.5499e-01, 2.1275e+00,  ..., 2.3257e-01, 3.7934e-01,
         1.1552e+01],
        [7.5823e-01, 7.4617e-05, 1.2352e-05,  ..., 1.0687e+00, 2.9321e-06,
         1.6589e+01],
        [5.1661e-01, 6.4545e-01, 4.6350e-01,  ..., 7.0831e-01, 4.4300e-02,
         7.3653e+00],
        ...,
        [3.4212e-01, 1.0389e+00, 3.9354e-01,  ..., 1.3082e+00, 1.3878e+00,
         6.5129e+00],
        [2.9431e-01, 1.0359e+00, 2.3197e+00,  ..., 8.1471e-01, 8.5662e-01,
         1.1298e+01],
        [9.9838e-03, 3.0187e-02, 9.4567e-03,  ..., 3.4208e-01, 3.7252e-02,
         7.4951e+00]], device='cuda:0', grad_fn=<SelectBackward>) tensor([7.1546e-02, 2.5499e-01, 2.1275e+00, 3.6478e+00, 3.4956e-01, 7.5791e+00,
        3.7338e+00, 2.4760e+00, 9.7236e-02, 7.3092e-01, 1.2517e+00, 1.7982e-01,
        6.9503e-01, 1.9633e-01, 6.1498e+00, 3.5204e-02, 9.2499e+00, 3.7632e-01,
        1.2236e+01, 4.3915e-01, 1.0228e+01, 1.1936e+01, 2.3103e-01, 1.5042e-07,
        1.2468e+01, 3.9618e-01, 3.7265e-01, 4.4669e+00, 4.5593e-01, 1.1535e+01,
        6.5056e-01, 2.2465e-01, 3.8978e-02, 3.5956e+00, 2.4666e+00, 9.3096e+00,
        5.7141e+00, 9.4356e+00, 1.0708e+01, 5.9201e-02, 3.2348e-01, 1.1139e+01,
        8.5903e-01, 1.1269e+01, 1.2459e+01, 3.7597e-01, 2.2571e+00, 1.3077e+01,
        1.4488e+00, 6.2625e+00, 1.0031e+01, 1.1818e+01, 1.1254e+01, 1.2067e+01,
        1.2150e+00, 9.7188e+00, 9.7453e+00, 6.2265e-01, 1.1585e-01, 1.6597e-01,
        7.6573e-02, 8.5276e-01, 9.3162e+00, 1.3360e+01, 6.6106e+00, 1.5939e+01,
        5.0089e-01, 8.3182e-02, 9.8871e+00, 2.1444e-01, 1.0322e+01, 8.8636e+00,
        9.6246e+00, 1.2396e+01, 2.7380e-01, 1.8133e-01, 2.4114e-01, 5.4410e-01,
        3.0842e+00, 1.3905e+01, 1.6951e-01, 6.6652e-02, 5.4724e+00, 7.5202e+00,
        3.9102e+00, 1.1578e+01, 8.4431e+00, 1.0053e+01, 9.0857e-02, 1.1869e+01,
        9.3042e+00, 7.6832e+00, 2.7051e-01, 1.3039e+01, 9.9403e+00, 1.0650e+01,
        3.2193e-03, 3.6699e-01, 5.3485e-01, 2.2431e+00, 3.2071e-01, 1.0315e+01,
        6.3739e-01, 6.0286e+00, 2.5378e-01, 6.7308e+00, 1.2618e+01, 6.9287e-01,
        1.1199e+01, 1.3058e+01, 1.4084e+01, 3.1295e+00, 2.7080e-01, 1.1911e+01,
        7.7342e-01, 8.1243e+00, 2.7944e-01, 1.0544e+01, 1.3942e+01, 6.2875e+00,
        6.2145e-01, 9.4341e-02, 1.1991e+01, 2.6494e+00, 1.3730e+01, 1.0207e+01,
        1.2620e+01, 1.7340e-03, 1.0203e+01, 1.1323e+00, 1.0697e+00, 9.7239e+00,
        9.3565e+00, 1.2396e+01, 2.6736e+00, 1.0713e+01, 3.4164e-01, 1.3162e+01,
        1.4486e+01, 2.1124e+00, 1.3259e-01, 1.0555e+01, 1.8042e-01, 1.4762e+01,
        2.3689e+00, 2.8511e-02, 1.0949e+01, 1.0446e+01, 6.6144e-01, 3.5320e+00,
        1.3718e+01, 1.3752e+01, 7.4599e-01, 5.4650e-01, 1.2728e+01, 1.2947e+01,
        1.0938e+01, 1.2502e+01, 1.1333e+01, 4.9764e+00, 5.6417e-01, 9.8614e+00,
        1.1186e+01, 5.1297e-01, 1.2090e+01, 3.2372e-01, 2.8036e+00, 1.8424e-01,
        3.0810e+00, 1.1666e+01, 5.7716e+00, 1.2602e+01, 1.2770e+01, 1.2007e+01,
        1.1341e+00, 4.2667e-01, 3.3429e-01, 5.0136e+00, 1.0452e+01, 1.2877e+01,
        1.2131e+01, 1.1507e-01, 3.2915e-01, 2.7084e-01, 1.1422e+01, 4.5190e-01,
        1.0660e-01, 6.0798e+00, 5.2745e-01, 1.3304e+01, 8.3589e+00, 7.3915e-02,
        7.8729e-01, 1.2611e+01, 1.1359e+00, 1.0383e-01, 5.0420e-02, 9.7468e+00,
        4.5168e-01, 1.2063e-01, 1.1364e+00, 1.5227e+01, 1.1231e+01, 9.8042e-01,
        1.2365e+01, 3.8811e-01, 2.3583e-01, 1.1805e-01, 5.3144e+00, 1.6154e+00,
        1.1092e+01, 2.9806e-01, 3.8424e+00, 1.0211e-01, 3.0679e+00, 6.7601e-02,
        3.5960e-01, 1.5492e+00, 5.5607e-02, 1.5031e+00, 1.5454e-01, 5.3906e-01,
        4.3316e-02, 1.5849e-01, 1.4443e+00, 1.5506e-01, 8.7235e+00, 6.7721e+00,
        8.5022e+00, 2.0400e-01, 6.6027e+00, 1.2527e+01, 1.0772e+01, 3.6900e+00,
        1.0032e+01, 3.4019e-01, 1.3049e-01, 7.3476e+00, 7.1734e+00, 1.0176e+01,
        7.4735e+00, 2.8553e-01, 8.9816e+00, 2.5665e-01, 9.0119e+00, 8.5746e-01,
        5.1585e-02, 1.1615e+01, 1.1736e+01, 1.9070e-01, 4.3190e-01, 1.5689e+01,
        1.0563e+01, 2.3257e-01, 3.7934e-01, 1.1552e+01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([[ 0.4830,  0.4695,  1.1998,  ...,  0.3432,  0.6737, 11.9207],
        [ 0.4594,  0.2792,  0.9778,  ...,  0.4311,  0.5357, 10.0467],
        [ 0.1911,  0.3797,  0.4181,  ...,  0.4096,  0.3525, 13.1689],
        ...,
        [ 0.4206,  0.4529,  0.9492,  ...,  0.3611,  0.4072,  9.1628],
        [ 0.4673,  0.4979,  0.9963,  ...,  0.5602,  0.4983,  9.8099],
        [ 0.4516,  0.4440,  1.1353,  ...,  0.5424,  0.7132, 11.6799]],
       device='cuda:0', grad_fn=<SqueezeBackward1>) tensor([ 0.4830,  0.4695,  1.1998,  2.2947,  0.8101, 10.0541,  3.2778,  1.8594,
         1.5356,  2.8011,  1.6960,  0.6583,  0.3095,  0.5394,  8.5447,  0.3952,
         5.7556,  0.2360, 11.9637,  0.5762, 12.1933, 13.7882,  0.2554,  0.6140,
        12.5522,  0.5728,  0.2555,  0.9903,  0.5652, 10.6606,  3.0743,  0.5426,
         0.5171,  1.6111,  1.1246,  8.3736,  1.4676,  3.4683, 12.7305,  0.2317,
         0.4294, 13.3485,  0.4910,  9.7709,  9.7947,  0.3333,  1.9894,  8.5802,
         1.3551,  3.1022,  7.9568, 14.6481, 10.7116, 11.9581,  1.0789, 13.0414,
        15.3283,  0.3481,  0.6017,  0.4445,  1.5863,  0.4654, 11.5328,  8.3095,
         8.4152, 15.5037,  0.4229,  0.4541,  6.0906,  2.1394, 13.7905,  3.6473,
        10.9672, 10.1814,  3.1663,  2.3458,  0.4554,  1.2225,  1.3424, 13.3163,
         0.3592,  0.7759, 11.0509, 11.3154,  0.9803, 12.9895,  6.5716,  9.5749,
         0.2803,  7.2080,  8.8505,  3.3401,  0.3662,  7.7956, 10.3975, 11.7606,
         0.2663,  3.4991,  0.2642,  1.2367,  0.3419, 11.3065,  3.0926,  8.1358,
         0.4199,  3.8252, 12.9111,  0.6670, 12.6381,  7.6974, 11.6182,  3.9997,
         0.2323, 14.0364,  0.3747,  4.7826,  1.9934, 11.5955, 14.4061,  7.4634,
         1.8127,  2.3517, 13.1681,  3.2279, 11.9200,  4.2678,  7.3987,  0.3400,
         3.6957,  0.5900,  0.5777,  7.0611, 14.4774, 11.3443,  0.7505,  7.9317,
         0.4814, 11.4329, 14.8380,  0.7919,  1.4378, 12.0064,  0.3986,  6.7073,
         1.3156,  3.3403, 10.7321,  4.5666,  0.7448,  0.6857, 10.6258, 13.8293,
         0.7055,  4.3550,  6.1626, 12.9631, 10.5125, 15.2288, 12.8916,  2.8188,
         2.0878, 13.5765,  9.6886,  2.0315, 11.8115,  3.5354,  1.6331,  0.2099,
         0.8322, 10.0643,  1.7130, 10.8202, 12.2787, 14.0088,  0.6465,  0.4533,
         0.2453,  1.8900, 11.7764,  7.3490, 11.3879,  2.3329,  0.2266,  0.2739,
        13.1855,  0.9019,  0.7307,  2.4849,  1.3030, 10.9149, 10.8227,  0.4465,
         0.1803, 13.9916,  0.7857,  0.2213,  0.5262,  8.0037,  0.8104,  0.4624,
         2.7932, 13.3634, 10.6947,  3.1661, 11.9500,  0.3448,  0.4569,  0.2737,
         5.9139,  0.8338,  9.6850,  0.6976,  1.3760,  0.4726,  1.1615,  0.5305,
         0.2909,  1.2452,  0.1572,  0.7066,  1.6227,  0.6221,  0.3901,  0.5797,
         0.4024,  0.5175, 10.1155,  4.4046,  7.2305,  0.4435,  2.8460, 12.6292,
         9.0243,  1.6184, 10.6367,  1.0472,  0.7312,  3.3511,  5.7311,  4.0282,
         7.5371,  2.5271,  4.8664,  0.2379,  6.1491,  0.3367,  0.8193,  9.9339,
         9.9888,  0.2930,  0.2343,  9.8122, 13.3678,  0.3432,  0.6737, 11.9207],
       device='cuda:0', grad_fn=<SelectBackward>)
mrr tensor([5.7010e-08, 4.8898e-06, 5.2213e-05, 1.0176e-04, 1.4006e-06, 9.9968e-01,
        5.5303e-06, 1.4127e-04, 4.8190e-09, 1.2046e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([7], dtype=torch.int32) tensor(4.8898e-06, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.7010e-08, 4.8898e-06, 5.2213e-05, 1.0176e-04, 1.4006e-06, 9.9968e-01,
        5.5303e-06, 1.4127e-04, 4.8190e-09, 1.2046e-05], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 7, 3, 2, 9, 6, 1, 4, 0, 8) tensor([0.6344])
auc tensor([5.7010e-08, 4.8898e-06, 5.2213e-05, 1.0176e-04, 1.4006e-06, 9.9968e-01,
        5.5303e-06, 1.4127e-04, 4.8190e-09, 1.2046e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.1875])
mrr tensor([2.6984e-25, 1.0000e+00, 1.2201e-28, 1.2886e-22, 5.7839e-24, 6.8521e-22,
        1.8962e-21, 3.4061e-23, 3.8517e-21, 7.2955e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.6984e-25, 1.0000e+00, 1.2201e-28, 1.2886e-22, 5.7839e-24, 6.8521e-22,
        1.8962e-21, 3.4061e-23, 3.8517e-21, 7.2955e-16], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 9, 8, 6, 5, 3, 7, 4, 0, 2) tensor([1.3010])
auc tensor([2.6984e-25, 1.0000e+00, 1.2201e-28, 1.2886e-22, 5.7839e-24, 6.8521e-22,
        1.8962e-21, 3.4061e-23, 3.8517e-21, 7.2955e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5625])
mrr tensor([9.4135e-11, 1.0000e+00, 1.1964e-08, 1.4322e-09, 1.3136e-13, 6.2417e-08,
        3.9012e-13, 8.4964e-11, 1.2189e-10, 2.0698e-10], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1.0000, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.4135e-11, 1.0000e+00, 1.1964e-08, 1.4322e-09, 1.3136e-13, 6.2417e-08,
        3.9012e-13, 8.4964e-11, 1.2189e-10, 2.0698e-10], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 5, 2, 3, 9, 8, 0, 7, 6, 4) tensor([1.3333])
auc tensor([9.4135e-11, 1.0000e+00, 1.1964e-08, 1.4322e-09, 1.3136e-13, 6.2417e-08,
        3.9012e-13, 8.4964e-11, 1.2189e-10, 2.0698e-10], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6875])
mrr tensor([2.8644e-09, 1.0000e+00, 5.2657e-22, 9.4685e-16, 1.0621e-12, 1.3086e-17,
        4.6296e-10, 6.2823e-19, 8.0404e-13, 9.5644e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.8644e-09, 1.0000e+00, 5.2657e-22, 9.4685e-16, 1.0621e-12, 1.3086e-17,
        4.6296e-10, 6.2823e-19, 8.0404e-13, 9.5644e-16], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 6, 4, 8, 9, 3, 5, 7, 2) tensor([1.6309])
auc tensor([2.8644e-09, 1.0000e+00, 5.2657e-22, 9.4685e-16, 1.0621e-12, 1.3086e-17,
        4.6296e-10, 6.2823e-19, 8.0404e-13, 9.5644e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([2.3055e-07, 9.2557e-01, 2.9548e-08, 1.9651e-06, 4.0380e-13, 2.2093e-09,
        3.7214e-02, 5.2040e-10, 3.7214e-02, 2.2821e-17], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9256, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.3055e-07, 9.2557e-01, 2.9548e-08, 1.9651e-06, 4.0380e-13, 2.2093e-09,
        3.7214e-02, 5.2040e-10, 3.7214e-02, 2.2821e-17], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 8, 6, 3, 0, 2, 5, 7, 4, 9) tensor([1.3869])
auc tensor([2.3055e-07, 9.2557e-01, 2.9548e-08, 1.9651e-06, 4.0380e-13, 2.2093e-09,
        3.7214e-02, 5.2040e-10, 3.7214e-02, 2.2821e-17], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8125])
mrr tensor([2.3349e-02, 9.7664e-01, 1.0935e-05, 4.9805e-09, 7.0196e-19, 5.7274e-10,
        4.3230e-10, 3.7932e-16, 6.2973e-10, 5.2019e-13], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9766, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.3349e-02, 9.7664e-01, 1.0935e-05, 4.9805e-09, 7.0196e-19, 5.7274e-10,
        4.3230e-10, 3.7932e-16, 6.2973e-10, 5.2019e-13], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 2, 3, 8, 5, 6, 9, 7, 4) tensor([1.6309])
auc tensor([2.3349e-02, 9.7664e-01, 1.0935e-05, 4.9805e-09, 7.0196e-19, 5.7274e-10,
        4.3230e-10, 3.7932e-16, 6.2973e-10, 5.2019e-13], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([1.7145e-16, 2.1548e-09, 4.8826e-17, 9.0366e-20, 8.1622e-11, 1.0000e+00,
        4.0578e-11, 3.5144e-19, 2.4854e-06, 8.1817e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(2.1548e-09, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.7145e-16, 2.1548e-09, 4.8826e-17, 9.0366e-20, 8.1622e-11, 1.0000e+00,
        4.0578e-11, 3.5144e-19, 2.4854e-06, 8.1817e-14], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 8, 1, 4, 6, 9, 0, 2, 7, 3) tensor([0.8333])
auc tensor([1.7145e-16, 2.1548e-09, 4.8826e-17, 9.0366e-20, 8.1622e-11, 1.0000e+00,
        4.0578e-11, 3.5144e-19, 2.4854e-06, 8.1817e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5625])
mrr tensor([2.2946e-03, 3.7555e-01, 2.9134e-01, 7.5108e-12, 2.2257e-03, 1.6131e-10,
        1.9437e-06, 3.2859e-01, 3.5733e-09, 6.9762e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.3756, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([2.2946e-03, 3.7555e-01, 2.9134e-01, 7.5108e-12, 2.2257e-03, 1.6131e-10,
        1.9437e-06, 3.2859e-01, 3.5733e-09, 6.9762e-07], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 7, 2, 0, 4, 6, 9, 8, 5, 3) tensor([1.4307])
auc tensor([2.2946e-03, 3.7555e-01, 2.9134e-01, 7.5108e-12, 2.2257e-03, 1.6131e-10,
        1.9437e-06, 3.2859e-01, 3.5733e-09, 6.9762e-07], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8750])
mrr tensor([5.9857e-03, 9.9401e-01, 7.9113e-18, 9.8060e-18, 5.1923e-28, 3.4714e-32,
        1.1689e-28, 7.0049e-26, 9.2172e-25, 7.8274e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9940, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.9857e-03, 9.9401e-01, 7.9113e-18, 9.8060e-18, 5.1923e-28, 3.4714e-32,
        1.1689e-28, 7.0049e-26, 9.2172e-25, 7.8274e-25], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 3, 2, 8, 9, 7, 4, 6, 5) tensor([1.6309])
auc tensor([5.9857e-03, 9.9401e-01, 7.9113e-18, 9.8060e-18, 5.1923e-28, 3.4714e-32,
        1.1689e-28, 7.0049e-26, 9.2172e-25, 7.8274e-25], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([4.9137e-08, 6.9008e-11, 9.6822e-01, 2.7303e-02, 2.7481e-13, 3.8421e-03,
        6.4889e-17, 6.2991e-04, 1.3283e-07, 1.0993e-11], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([6], dtype=torch.int32) tensor(4.9137e-08, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([4.9137e-08, 6.9008e-11, 9.6822e-01, 2.7303e-02, 2.7481e-13, 3.8421e-03,
        6.4889e-17, 6.2991e-04, 1.3283e-07, 1.0993e-11], device='cuda:0',
       grad_fn=<SelectBackward>) (2, 3, 5, 7, 8, 0, 1, 9, 4, 6) tensor([0.6895])
auc tensor([4.9137e-08, 6.9008e-11, 9.6822e-01, 2.7303e-02, 2.7481e-13, 3.8421e-03,
        6.4889e-17, 6.2991e-04, 1.3283e-07, 1.0993e-11], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([1.0000e+00, 2.6062e-07, 4.7253e-13, 1.3496e-22, 4.5588e-13, 2.1656e-23,
        3.9056e-19, 1.7238e-20, 3.2851e-22, 6.3106e-20], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1.0000, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 2.6062e-07, 4.7253e-13, 1.3496e-22, 4.5588e-13, 2.1656e-23,
        3.9056e-19, 1.7238e-20, 3.2851e-22, 6.3106e-20], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 2, 4, 6, 9, 7, 8, 3, 5) tensor([1.6309])
auc tensor([1.0000e+00, 2.6062e-07, 4.7253e-13, 1.3496e-22, 4.5588e-13, 2.1656e-23,
        3.9056e-19, 1.7238e-20, 3.2851e-22, 6.3106e-20], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([1.0000e+00, 1.6302e-20, 7.1317e-18, 6.7170e-09, 1.6165e-16, 6.3456e-15,
        1.9520e-21, 4.2941e-19, 6.0148e-09, 1.4882e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 1.6302e-20, 7.1317e-18, 6.7170e-09, 1.6165e-16, 6.3456e-15,
        1.9520e-21, 4.2941e-19, 6.0148e-09, 1.4882e-16], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 3, 8, 5, 4, 9, 2, 7, 1, 6) tensor([1.3010])
auc tensor([1.0000e+00, 1.6302e-20, 7.1317e-18, 6.7170e-09, 1.6165e-16, 6.3456e-15,
        1.9520e-21, 4.2941e-19, 6.0148e-09, 1.4882e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5625])
mrr tensor([6.5997e-09, 1.5943e-03, 1.4493e-09, 7.0368e-06, 2.8766e-03, 8.2380e-09,
        9.5743e-07, 1.0356e-08, 1.3155e-06, 9.9552e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(0.0016, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([6.5997e-09, 1.5943e-03, 1.4493e-09, 7.0368e-06, 2.8766e-03, 8.2380e-09,
        9.5743e-07, 1.0356e-08, 1.3155e-06, 9.9552e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (9, 4, 1, 3, 8, 6, 7, 5, 0, 2) tensor([0.8010])
auc tensor([6.5997e-09, 1.5943e-03, 1.4493e-09, 7.0368e-06, 2.8766e-03, 8.2380e-09,
        9.5743e-07, 1.0356e-08, 1.3155e-06, 9.9552e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([5.7433e-25, 4.0300e-25, 9.8243e-20, 1.0000e+00, 4.3475e-20, 1.6708e-21,
        7.1911e-22, 2.9148e-23, 1.3412e-15, 2.6221e-22], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([9], dtype=torch.int32) tensor(5.7433e-25, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.7433e-25, 4.0300e-25, 9.8243e-20, 1.0000e+00, 4.3475e-20, 1.6708e-21,
        7.1911e-22, 2.9148e-23, 1.3412e-15, 2.6221e-22], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 8, 2, 4, 5, 6, 9, 7, 0, 1) tensor([0.5901])
auc tensor([5.7433e-25, 4.0300e-25, 9.8243e-20, 1.0000e+00, 4.3475e-20, 1.6708e-21,
        7.1911e-22, 2.9148e-23, 1.3412e-15, 2.6221e-22], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.])
mrr tensor([1.9283e-07, 3.5753e-07, 3.5930e-06, 9.4569e-10, 1.9120e-09, 1.3624e-08,
        2.6037e-12, 9.9979e-01, 2.0265e-04, 4.8805e-15], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([4], dtype=torch.int32) tensor(3.5753e-07, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.9283e-07, 3.5753e-07, 3.5930e-06, 9.4569e-10, 1.9120e-09, 1.3624e-08,
        2.6037e-12, 9.9979e-01, 2.0265e-04, 4.8805e-15], device='cuda:0',
       grad_fn=<SelectBackward>) (7, 8, 2, 1, 0, 5, 4, 3, 6, 9) tensor([0.8175])
auc tensor([1.9283e-07, 3.5753e-07, 3.5930e-06, 9.4569e-10, 1.9120e-09, 1.3624e-08,
        2.6037e-12, 9.9979e-01, 2.0265e-04, 4.8805e-15], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([3.3954e-15, 1.0789e-09, 6.3314e-11, 3.9224e-14, 1.9986e-11, 8.1090e-12,
        1.1179e-11, 1.1045e-07, 1.0000e+00, 7.6652e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(1.0789e-09, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.3954e-15, 1.0789e-09, 6.3314e-11, 3.9224e-14, 1.9986e-11, 8.1090e-12,
        1.1179e-11, 1.1045e-07, 1.0000e+00, 7.6652e-16], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 7, 1, 2, 4, 6, 5, 3, 0, 9) tensor([0.8010])
auc tensor([3.3954e-15, 1.0789e-09, 6.3314e-11, 3.9224e-14, 1.9986e-11, 8.1090e-12,
        1.1179e-11, 1.1045e-07, 1.0000e+00, 7.6652e-16], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.4375])
mrr tensor([7.1567e-01, 6.6107e-03, 1.5701e-02, 4.3242e-05, 6.3061e-09, 6.6809e-05,
        1.9280e-02, 6.2905e-09, 1.4678e-02, 2.2795e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.7157, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([7.1567e-01, 6.6107e-03, 1.5701e-02, 4.3242e-05, 6.3061e-09, 6.6809e-05,
        1.9280e-02, 6.2905e-09, 1.4678e-02, 2.2795e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 9, 6, 2, 8, 1, 5, 3, 4, 7) tensor([1.3562])
auc tensor([7.1567e-01, 6.6107e-03, 1.5701e-02, 4.3242e-05, 6.3061e-09, 6.6809e-05,
        1.9280e-02, 6.2905e-09, 1.4678e-02, 2.2795e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([9.9240e-01, 8.6136e-06, 9.8578e-08, 1.7385e-07, 7.5947e-03, 5.6856e-13,
        3.0185e-09, 4.6778e-10, 5.8421e-13, 1.3245e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9924, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.9240e-01, 8.6136e-06, 9.8578e-08, 1.7385e-07, 7.5947e-03, 5.6856e-13,
        3.0185e-09, 4.6778e-10, 5.8421e-13, 1.3245e-06], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 4, 1, 9, 3, 2, 6, 7, 8, 5) tensor([1.5000])
auc tensor([9.9240e-01, 8.6136e-06, 9.8578e-08, 1.7385e-07, 7.5947e-03, 5.6856e-13,
        3.0185e-09, 4.6778e-10, 5.8421e-13, 1.3245e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.9375])
mrr tensor([1.0000e+00, 7.3001e-32, 4.1235e-32, 6.9903e-22, 8.3244e-26, 1.1963e-33,
        4.6728e-30, 2.9919e-16, 5.6371e-20, 3.0015e-24], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 7.3001e-32, 4.1235e-32, 6.9903e-22, 8.3244e-26, 1.1963e-33,
        4.6728e-30, 2.9919e-16, 5.6371e-20, 3.0015e-24], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 7, 8, 3, 9, 4, 6, 1, 2, 5) tensor([1.3155])
auc tensor([1.0000e+00, 7.3001e-32, 4.1235e-32, 6.9903e-22, 8.3244e-26, 1.1963e-33,
        4.6728e-30, 2.9919e-16, 5.6371e-20, 3.0015e-24], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([7.6649e-01, 2.8633e-08, 9.6692e-08, 1.5923e-15, 3.9467e-05, 1.8332e-12,
        3.0993e-13, 1.5744e-08, 2.3347e-01, 6.8898e-11], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.7665, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([7.6649e-01, 2.8633e-08, 9.6692e-08, 1.5923e-15, 3.9467e-05, 1.8332e-12,
        3.0993e-13, 1.5744e-08, 2.3347e-01, 6.8898e-11], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 8, 4, 2, 1, 7, 9, 5, 6, 3) tensor([1.3869])
auc tensor([7.6649e-01, 2.8633e-08, 9.6692e-08, 1.5923e-15, 3.9467e-05, 1.8332e-12,
        3.0993e-13, 1.5744e-08, 2.3347e-01, 6.8898e-11], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.8125])
mrr tensor([4.3933e-13, 1.1750e-20, 5.3114e-04, 9.8818e-01, 3.3155e-09, 1.1563e-14,
        5.2684e-03, 2.6289e-12, 3.9829e-10, 6.0214e-03], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([8], dtype=torch.int32) tensor(4.3933e-13, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([4.3933e-13, 1.1750e-20, 5.3114e-04, 9.8818e-01, 3.3155e-09, 1.1563e-14,
        5.2684e-03, 2.6289e-12, 3.9829e-10, 6.0214e-03], device='cuda:0',
       grad_fn=<SelectBackward>) (3, 9, 6, 2, 4, 8, 7, 0, 5, 1) tensor([0.6045])
auc tensor([4.3933e-13, 1.1750e-20, 5.3114e-04, 9.8818e-01, 3.3155e-09, 1.1563e-14,
        5.2684e-03, 2.6289e-12, 3.9829e-10, 6.0214e-03], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.0625])
mrr tensor([1.0597e-07, 3.0555e-07, 1.1583e-02, 1.1229e-13, 9.8842e-01, 1.8238e-08,
        1.2511e-14, 7.9049e-17, 1.0033e-10, 4.2151e-09], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(3.0555e-07, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0597e-07, 3.0555e-07, 1.1583e-02, 1.1229e-13, 9.8842e-01, 1.8238e-08,
        1.2511e-14, 7.9049e-17, 1.0033e-10, 4.2151e-09], device='cuda:0',
       grad_fn=<SelectBackward>) (4, 2, 1, 0, 5, 9, 8, 3, 6, 7) tensor([0.9307])
auc tensor([1.0597e-07, 3.0555e-07, 1.1583e-02, 1.1229e-13, 9.8842e-01, 1.8238e-08,
        1.2511e-14, 7.9049e-17, 1.0033e-10, 4.2151e-09], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([9.7861e-01, 2.1388e-02, 2.1044e-12, 1.5598e-13, 2.1485e-11, 1.0590e-06,
        2.0460e-16, 4.1183e-10, 9.3474e-11, 4.4604e-13], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9786, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.7861e-01, 2.1388e-02, 2.1044e-12, 1.5598e-13, 2.1485e-11, 1.0590e-06,
        2.0460e-16, 4.1183e-10, 9.3474e-11, 4.4604e-13], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 5, 7, 8, 4, 2, 9, 3, 6) tensor([1.6309])
auc tensor([9.7861e-01, 2.1388e-02, 2.1044e-12, 1.5598e-13, 2.1485e-11, 1.0590e-06,
        2.0460e-16, 4.1183e-10, 9.3474e-11, 4.4604e-13], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([3.0289e-22, 9.9961e-01, 1.9075e-14, 3.8131e-11, 6.2551e-12, 2.4574e-10,
        3.8681e-04, 1.6561e-15, 5.1707e-16, 2.0387e-15], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.9996, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([3.0289e-22, 9.9961e-01, 1.9075e-14, 3.8131e-11, 6.2551e-12, 2.4574e-10,
        3.8681e-04, 1.6561e-15, 5.1707e-16, 2.0387e-15], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 5, 3, 4, 2, 9, 7, 8, 0) tensor([1.2891])
auc tensor([3.0289e-22, 9.9961e-01, 1.9075e-14, 3.8131e-11, 6.2551e-12, 2.4574e-10,
        3.8681e-04, 1.6561e-15, 5.1707e-16, 2.0387e-15], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.5000])
mrr tensor([6.0771e-11, 5.2503e-01, 2.0284e-10, 2.1142e-02, 1.4224e-11, 1.3963e-11,
        4.5230e-01, 1.9284e-13, 1.5277e-03, 8.3652e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.5250, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([6.0771e-11, 5.2503e-01, 2.0284e-10, 2.1142e-02, 1.4224e-11, 1.3963e-11,
        4.5230e-01, 1.9284e-13, 1.5277e-03, 8.3652e-14], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 3, 8, 2, 0, 4, 5, 7, 9) tensor([1.3562])
auc tensor([6.0771e-11, 5.2503e-01, 2.0284e-10, 2.1142e-02, 1.4224e-11, 1.3963e-11,
        4.5230e-01, 1.9284e-13, 1.5277e-03, 8.3652e-14], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.7500])
mrr tensor([8.0825e-16, 1.0000e+00, 7.4485e-24, 1.3202e-34, 1.5859e-24, 4.6980e-19,
        7.1952e-19, 8.8208e-23, 1.9339e-19, 9.0812e-33], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([8.0825e-16, 1.0000e+00, 7.4485e-24, 1.3202e-34, 1.5859e-24, 4.6980e-19,
        7.1952e-19, 8.8208e-23, 1.9339e-19, 9.0812e-33], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 0, 6, 5, 8, 7, 2, 4, 9, 3) tensor([1.6309])
auc tensor([8.0825e-16, 1.0000e+00, 7.4485e-24, 1.3202e-34, 1.5859e-24, 4.6980e-19,
        7.1952e-19, 8.8208e-23, 1.9339e-19, 9.0812e-33], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([5.1345e-12, 3.3829e-09, 1.0171e-04, 1.5726e-07, 1.8710e-10, 9.0854e-17,
        8.9052e-09, 1.0058e-09, 3.3554e-12, 9.9990e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([5], dtype=torch.int32) tensor(3.3829e-09, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([5.1345e-12, 3.3829e-09, 1.0171e-04, 1.5726e-07, 1.8710e-10, 9.0854e-17,
        8.9052e-09, 1.0058e-09, 3.3554e-12, 9.9990e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (9, 2, 3, 6, 1, 7, 4, 0, 8, 5) tensor([0.7023])
auc tensor([5.1345e-12, 3.3829e-09, 1.0171e-04, 1.5726e-07, 1.8710e-10, 9.0854e-17,
        8.9052e-09, 1.0058e-09, 3.3554e-12, 9.9990e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.3750])
mrr tensor([1.0000e+00, 2.8877e-12, 4.0227e-17, 3.7338e-20, 4.4076e-16, 2.1921e-14,
        2.7687e-21, 1.0910e-20, 1.9001e-14, 7.0331e-22], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(1., device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([1.0000e+00, 2.8877e-12, 4.0227e-17, 3.7338e-20, 4.4076e-16, 2.1921e-14,
        2.7687e-21, 1.0910e-20, 1.9001e-14, 7.0331e-22], device='cuda:0',
       grad_fn=<SelectBackward>) (0, 1, 5, 8, 4, 2, 3, 7, 6, 9) tensor([1.6309])
auc tensor([1.0000e+00, 2.8877e-12, 4.0227e-17, 3.7338e-20, 4.4076e-16, 2.1921e-14,
        2.7687e-21, 1.0910e-20, 1.9001e-14, 7.0331e-22], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1.])
mrr tensor([8.8905e-15, 1.8550e-05, 2.0096e-17, 1.2851e-09, 9.2578e-16, 1.2114e-18,
        3.3515e-07, 3.9245e-15, 9.9996e-01, 2.4414e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([3], dtype=torch.int32) tensor(1.8550e-05, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([8.8905e-15, 1.8550e-05, 2.0096e-17, 1.2851e-09, 9.2578e-16, 1.2114e-18,
        3.3515e-07, 3.9245e-15, 9.9996e-01, 2.4414e-05], device='cuda:0',
       grad_fn=<SelectBackward>) (8, 9, 1, 6, 3, 0, 7, 4, 2, 5) tensor([0.8562])
auc tensor([8.8905e-15, 1.8550e-05, 2.0096e-17, 1.2851e-09, 9.2578e-16, 1.2114e-18,
        3.3515e-07, 3.9245e-15, 9.9996e-01, 2.4414e-05], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6250])
mrr tensor([4.9574e-07, 6.1243e-01, 1.7887e-03, 1.3110e-05, 1.8189e-12, 3.0391e-06,
        3.8576e-01, 4.9777e-14, 7.3427e-08, 1.7184e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([1], dtype=torch.int32) tensor(0.6124, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([4.9574e-07, 6.1243e-01, 1.7887e-03, 1.3110e-05, 1.8189e-12, 3.0391e-06,
        3.8576e-01, 4.9777e-14, 7.3427e-08, 1.7184e-06], device='cuda:0',
       grad_fn=<SelectBackward>) (1, 6, 2, 3, 5, 9, 0, 8, 4, 7) tensor([1.3333])
auc tensor([4.9574e-07, 6.1243e-01, 1.7887e-03, 1.3110e-05, 1.8189e-12, 3.0391e-06,
        3.8576e-01, 4.9777e-14, 7.3427e-08, 1.7184e-06], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.6875])
mrr tensor([9.1344e-10, 6.3644e-11, 1.0236e-06, 3.1399e-08, 3.7180e-05, 6.1692e-01,
        2.0005e-07, 6.0172e-12, 3.2402e-04, 3.8272e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([8], dtype=torch.int32) tensor(9.1344e-10, device='cuda:0', grad_fn=<SelectBackward>)
acc tensor([9.1344e-10, 6.3644e-11, 1.0236e-06, 3.1399e-08, 3.7180e-05, 6.1692e-01,
        2.0005e-07, 6.0172e-12, 3.2402e-04, 3.8272e-01], device='cuda:0',
       grad_fn=<SelectBackward>) (5, 9, 8, 4, 2, 6, 3, 0, 1, 7) tensor([0.6165])
auc tensor([9.1344e-10, 6.3644e-11, 1.0236e-06, 3.1399e-08, 3.7180e-05, 6.1692e-01,
        2.0005e-07, 6.0172e-12, 3.2402e-04, 3.8272e-01], device='cuda:0',
       grad_fn=<SelectBackward>) tensor([0.1250])

Evaluation - loss: 434.880005  dcg: 66.5526 mrr: 61.5396 auc: 60.4083
Saving best model, acc: 66.5526%

Batch[700] - loss: 422.395203  dcg: 0.6586 mrr: 0.6531 auc: 0.5352
Evaluation - loss: 434.599731  dcg: 66.6395 mrr: 61.6759 auc: 60.7926
Saving best model, acc: 66.6395%

Batch[800] - loss: 442.498993  dcg: 0.6456 mrr: 0.6001 auc: 0.5684
Evaluation - loss: 432.653076  dcg: 67.1752 mrr: 62.0052 auc: 61.0131
Saving best model, acc: 67.1752%

Batch[900] - loss: 440.607269  dcg: 0.6668 mrr: 0.6165 auc: 0.6113
Evaluation - loss: 427.272705  dcg: 67.8041 mrr: 63.1923 auc: 61.6368
Saving best model, acc: 67.8041%

Batch[1000] - loss: 464.338776  dcg: 0.6330 mrr: 0.5658 auc: 0.5586
Evaluation - loss: 426.164062  dcg: 67.3493 mrr: 63.4853 auc: 60.4398
Batch[1100] - loss: 439.179291  dcg: 0.6512 mrr: 0.6177 auc: 0.5918
Evaluation - loss: 428.167511  dcg: 67.0721 mrr: 63.2227 auc: 59.7152
Batch[1200] - loss: 474.298248  dcg: 0.6086 mrr: 0.5497 auc: 0.4980
Evaluation - loss: 431.183380  dcg: 66.6074 mrr: 63.1319 auc: 58.2724
Batch[1300] - loss: 395.964905  dcg: 0.7263 mrr: 0.7281 auc: 0.6387
Evaluation - loss: 427.251434  dcg: 66.7435 mrr: 63.3019 auc: 57.8125
Batch[1400] - loss: 370.735535  dcg: 0.7575 mrr: 0.7763 auc: 0.6914
Evaluation - loss: 429.637024  dcg: 66.4652 mrr: 63.3763 auc: 57.0502
Batch[1500] - loss: 372.195648  dcg: 0.7309 mrr: 0.7453 auc: 0.6348
Evaluation - loss: 435.873718  dcg: 65.9966 mrr: 61.9751 auc: 58.0645
Batch[1600] - loss: 474.331329  dcg: 0.6098 mrr: 0.5314 auc: 0.5195
Evaluation - loss: 432.354584  dcg: 66.2496 mrr: 62.7604 auc: 58.0330
Batch[1700] - loss: 426.062988  dcg: 0.6676 mrr: 0.6430 auc: 0.5859
Evaluation - loss: 427.806305  dcg: 66.6448 mrr: 63.8122 auc: 57.4156
Batch[1800] - loss: 413.432251  dcg: 0.6927 mrr: 0.6719 auc: 0.6094
Evaluation - loss: 437.742004  dcg: 65.2611 mrr: 61.5136 auc: 55.6515
Batch[1900] - loss: 449.650757  dcg: 0.6296 mrr: 0.5900 auc: 0.5215
Evaluation - loss: 433.197784  dcg: 65.2385 mrr: 62.7061 auc: 54.2024

early stop by 1000 steps, acc: 67.8041%
Exiting from training early
(pytorch) user10@430:~/lyx/src1$
Network error: Software caused connection abort

───────────────────────────────────────────────────────────────────────────────────────────────────────────────

Session stopped
    - Press <return> to exit tab
    - Press R to restart session
    - Press S to save terminal output to file


early stop by 1000 steps, acc: 68.2971%
