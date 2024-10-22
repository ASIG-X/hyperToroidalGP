# hyperToroidalGP
This repository contains the implementation of defining Gaussian processes on hypertoroidal manifolds. Detailed information can be found in our paper on [arXiv](https://arxiv.org/abs/2303.06799).

The work has been publlished on 2024 European Control Conference (ECC).
## BibTex Citation
Thank you for citing our paper if you use any of this code:
```
@InProceedings{ECC24_Cao,
  title={Gaussian Process on the Product of Directional Manifolds},
  author={Cao, Ziyu and Li, Kailai},
  booktitle={Proceedings of the 2024 European Control Conference},
  month={June},
  year={2024}
}
```
## Dependencies
* [Manopt](https://www.manopt.org/)
* [libDirectional](https://github.com/KIT-ISAS/libDirectional)
## Usage
* For the example of the Gaussian process on the unit circle, run the script `examples/GPonCircleExample.m`.
* For the example of the Gaussian process on the hypertorus, first run the script `examples/recursiveLocalizationTraining.m` and then `examples/recursiveLocalizationTest.m` for training data and testing, respectively.
## Contributors
Ziyu Cao (email: ziyu.cao@liu.se)

Kailai Li (email: kailai.li@rug.nl)
## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

We are constantly working on improving our code. For any technical issues, please contact Ziyu Cao (ziyu.cao@liu.se).
