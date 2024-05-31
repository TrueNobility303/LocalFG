# LocalFG

[ICML23] Can Forward Gradient Match Backpropagation?


## Algorithm
Code base: https://github.com/streethagore/ForwardLocalGradient.

Two algorithms are implemented with Pytorch.

* LocalFG-W: Forward gradient with weight perturbation, guided with local guess.
* LocalFG-A: Forward gradient with activity perturbation, guided with local guess.

FG-W and FG-A are proposed in [1], while local guess guidance is proposed in [2].

[1] Mengye Ren, Simon Kornblith, Renjie Liao, and Geoffrey Hinton. "Scaling Forward Gradient With Local Losses." In ICLR, 2023.

[2] Louis Fournier , Stéphane Rivaud, Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon. "Can Forward Gradient Match Backpropagation?" In ICML, 2023.

## Reproduce the results

Run the commands with

```
python -u main.py --arch resnet18-16b --n_epoch 100 --lr 5e-2 --wd 5e-4 --algo LocalFG-W
python -u main.py --arch resnet18-16b --n_epoch 100 --lr 5e-2 --wd 5e-4 --algo LocalFG-A
```

will give the following test accuracies on CIFAR10 dataset.

| Algorithm      | LocalFG-W | LocalFG-A |
| -------------- | --------- | --------- |
| Train Accuracy | 97.9 %    | 98.2 %    |
| Test Accuracy  | 88.8 %    | 89.4 %    |

In our experiments, activity perturbation has better performance than weight pertubation, as also observed in [1].

