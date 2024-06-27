# hyperToroidalGP
This repository contains the implementation of defining Gaussian processes on hypertoroidal manifolds. Detailed information can be found in our paper on [arXiv](https://arxiv.org/abs/2303.06799).

The work has been publlished on 2024 European Control Conference (ECC).
## Dependencies
* [Manopt](https://www.manopt.org/)
* [libDirectional](https://github.com/KIT-ISAS/libDirectional)
## Usage
* For the example of the Gaussian process on the unit circle, run the script `GPonCircleExample.m`.
* For the example of the Gaussian process on the hypertorus, first run the script `recursiveLocalizationTraining.m` and then `recursiveLocalizationTest.m` for training data and testing, respectively.
## Contributors
Ziyu Cao (email: ziyu.cao@liu.se)

Kailai Li (email: kailai.li@rug.nl)
## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

We are constantly working on improving our code. For any technical issues, please contact Ziyu Cao (ziyu.cao@liu.se).
