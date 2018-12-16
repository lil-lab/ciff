# Cornell Instruction Following Framework (CIFF)

CIFF is intended to provide an integrated framework for developing and experimenting with various natural language instruction following framework. Currently it provides a common interface for 3 datasets and simulators, several models and learning algorithm.

**Contents of this repository**

Contains simulators and dataset for 3 domains for natural language instruction following: 

     a) Block World Dataset (Bisk et al. 2016): A fully-observed map for moving blocks around on a map.

     b) LANI (Misra et al. 2018): A partially observed domain for flying drone in an open space.

     c) CHAI (Misra et al. 2018): A partially observed domain for navigation and simple manipulation in a 3D house.

The code contains experiments for training and testing various models and baselines. Includes:

      a) Simple baselines like stop, random baseline, 

      b) Models like Misra et al. 2017, Gated Attention Chaplot et al. 2017, etc.
        
      c) Training algorithms like behaviour cloning, A3C, Reinforce.

**Dataset and Simulators:** Dataset and simulators are available here: http://clic.nlp.cornell.edu/resources/Misra-EMNLP-2018/
   
 Test scripts for visualizing the dataset and simulators will be available on the wiki soon.

**Code To Come Soon:** We will release a beta version soon.

**Credits**

Maintained by: Dipendra Misra (dkm@cs.cornell.edu)

Researchers and Developers: Dipendra Misra, Andrew Bennett, Max Shatkin, Eyvind Niklasson, Valts Blukis, and Yoav Artzi

**Publications**

Publications using models, data or simulators provided with CIFF.

1) CoRL submission

2) Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction *Dipendra Misra, Andrew Bennett, Valts Blukis, Eyvind Niklasson, Max Shatkhin, and Yoav Artzi*, [EMNLP 2018](https://arxiv.org/abs/1809.00786)

3) Scheduled Policy Optimization for Natural Language Communication with Intelligent Agents, *Wenhan Xiong, Xiaoxiao Guo, Mo Yu, Shiyu Chang, Bowen Zhou, William Yang Wang*, [arXiv 2018](https://arxiv.org/abs/1806.06187)

4) Reinforcement Learning for Mapping Instructions to Actions with Reward Learning, *Dipendra Misra and Yoav Artzi*, AAAI Fall Symposium on Natural Language Communication for Human Robot Interaction. [Paper](http://www.ttic.edu/nchrc/papers/19.pdf)

5) Mapping Instructions and Visual Observations to Actions with Reinforcement Learning, *Dipendra Misra, John Langford and Yoav Artzi*, EMNLP 2017. [Paper](http://www.cs.cornell.edu/~dkm/papers/mla-emnlp.2017.pdf)

**How to Use**

1) Clone the repostory using `git clone https://github.com/clic-lab/instruction-following-framework.git`

2) Repository only contains the source code. You will have to download the data and simulators from Cornell Box. 
https://cornell.app.box.com/folder/48719193956. Download the data and simulator folder using rclone inside the directory where your code is. (If you don't have access to Cornell Box folder, please contact Dipendra Misra at dkm@cs.cornell.edu).

`rclone sync cornellbox:/data_and_simulators/data data`

`rclone sync cornellbox:/data_and_simulators/simulators simulators`

Your directory structure should look like:

- Main directory:

     - data
     
     - simulators
     
     - src
     
3) Run an experiment. See wiki for more details.
