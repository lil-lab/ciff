# Cornell Instruction Following Framework (CIFF)

CIFF is intended to provide an integrated framework for developing and experimenting with various natural language instruction following framework. Currently it provides a common interface for 3 datasets and simulators, several models and learning algorithm.

**Contents of this repository**

Contains simulators and dataset for 3 domains for natural language instruction following: 

     a) Block World Dataset (Bisk et al. 2016): A fully-observed map for moving blocks around on a map.

     b) SLUG (Bennett et al. 2017): A partially observed problem for flying drone in a open space.

     c) CHALET (Misra et al. 2017): A partially observed problem for navigation and manipulation in a 3D house

The code contains experiments for training and testing various models and baselines. Includes:

      a) Simple baselines like stop, random baseline, most frequent action.

      b) Models like Misra et al. 2017, Gated Attention Chaplot et al. 2018, etc.
        
      c) Training algorithms like behaviour cloning, A3C, Reinforce.

**Code To Come Soon:** We will release a beta version soon.

**Credits**

Maintained by: Dipendra Misra (dkm@cs.cornell.edu)

Researchers and Developers: Dipendra Misra, Andrew Bennett, Max Shatkin, Eyvind Nikalson, Valts Blukis, and Yoav Artzi

**Publications**

Submissions using models, data or simulators provided with CIFF.

1) Mapping Navigation Instructions to Continuous Control Actions with Position Visitation Prediction *Valts Blukis, Dipendra Misra, Ross A. Knepper, and Yoav Artzi*, [CoRL 2018]  (uses the LANI dataset)

2) Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction *Dipendra Misra, Andrew Bennett, Valts Blukis, Eyvind Niklasson, Max Shatkhin, and Yoav Artzi*, [EMNLP 2018]

3) Scheduled Policy Optimization for Natural Language Communication with Intelligent Agents, *Wenhan Xiong, Xiaoxiao Guo, Mo Yu, Shiyu Chang, Bowen Zhou, William Yang Wang*, [arXiv 2018](https://arxiv.org/abs/1806.06187)

4) Reinforcement Learning for Mapping Instructions to Actions with Reward Learning, *Dipendra Misra and Yoav Artzi*, AAAI Fall Symposium on Natural Language Communication for Human Robot Interaction. [Paper](http://www.ttic.edu/nchrc/papers/19.pdf)

5) Mapping Instructions and Visual Observations to Actions with Reinforcement Learning, *Dipendra Misra, John Langford and Yoav Artzi*, EMNLP 2017. [Paper](http://www.cs.cornell.edu/~dkm/papers/mla-emnlp.2017.pdf)
