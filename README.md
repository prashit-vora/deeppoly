# deeppoly-cpp

Adversarial robustness certification for neural networks, in C++.

Trains a fully connected network on MNIST, attacks it with FGSM, then formally
certifies per-image robustness using the DeepPoly abstract domain from
Singh et al., POPL 2019.


---

## how it works

The verifier propagates a symbolic abstract element through the network
instead of running it on every possible perturbed input (exponentially many).

Each neuron tracks two linear bounds over the input pixels plus concrete
interval bounds. Three transformers do the work:

**Affine** — exact symbolic propagation through linear layers via backsubstitution
to input variables.

**ReLU** — three cases: zero (exact), identity (exact), mixed (convex hull
approximation with minimum area lower bound).

**Sigmoid** — monotone, so concrete bounds are sigmoid applied to the pre-activation
bounds directly.

If the symbolic lower bound on the true class output exceeds the upper bound of
every other class across the entire L∞ ball around the input, the network is
certified robust. No counterexample exists.

---

## stack

```
C++20        neural network + verifier
tiny-cpp-nn  header-only training library
Raylib 5.5   GPU-accelerated visualization window
CMake        build system, fetches dependencies automatically
```

---

## build

```bash
git clone https://github.com/jeetrex17/deeppoly-cpp
cd deeppoly-cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

---

## run

```bash
cd build/data && bash ../../data/download.sh && cd ..

./train  ../data
./attack ../data 100 0.10
./verify 0.05
./show
```

---

## architecture

```
include/mnist_loader.hpp    IDX binary parser
include/fgsm.hpp            one-step gradient sign attack
include/deeppoly.hpp        abstract domain with affine + ReLU transformers
include/results.hpp         binary serialization
src/train.cpp               784 → 128 → 64 → 10, saves models/mnist.bin
src/attack.cpp              FGSM on N images, saves output/results.bin
src/verify.cpp              DeepPoly certification, updates results
src/show.cpp                Raylib window, scrollable card grid
```

---

## Python adversarial training

A Python implementation of PGD adversarial training (Madry et al. 2018)
with DeepPoly certification for a CNN on MNIST.

**Architecture:** `Conv(1→16,4,s2)→ReLU→Conv(16→32,4,s2)→ReLU→Dense(1568→64)→ReLU→Dense(64→10)`

**Min-max game:** Defender (θ) trains against an Attacker (δ) that maximises
loss via PGD. Per-batch PGD-7 during training, PGD-40 for evaluation.
DeepPoly certification provides sound per-image robustness guarantees.

### requirements

```
torch, numpy, matplotlib
```

### usage

```bash
python3 adversarial_game.py <n_train> <n_test> <epsilon>
```

Defaults: 5000 train / 200 test / ε=0.10. Outputs accuracy, PGD-40 attack
success, DeepPoly certified robustness, and a plot saved to
`training_results.png`.

### files

```
adversarial_game.py   CNN + PGD training + evaluation + DeepPoly verifier
training_results.png  saved plot (attack curve + test breakdown)
```

---

## reference

Gagandeep Singh, Timon Gehr, Markus Püschel, Martin Vechev.
*An Abstract Domain for Certifying Neural Networks.*
POPL 2019. https://doi.org/10.1145/3290354

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu.
*Towards Deep Learning Models Resistant to Adversarial Attacks.*
ICLR 2018. https://arxiv.org/abs/1706.06083
