# Swallow

This is the source code for paper "Swallow: A Transfer-Robust Website Fingerprinting Attack via Consistent Feature Learning", accepted in ACM CCS 2025.

# Dataset
## Our dataset

| Dataset ID | Location     | Browser | Time    | Size       |
|------------|--------------|---------|---------|------------|
| $D_1$      | Chicago      | TBB     | 2024-7  | 100 × 100  |
| $D_2$      | Singapore    | TBB     | 2024-7  | 100 × 100  |
| $D_3$      | London       | TBB     | 2024-7  | 100 × 100  |
| $D_4$      | Johannesburg | TBB     | 2024-7  | 100 × 100  |
| $D_5$      | Mumbai       | TBB     | 2024-7  | 100 × 100  |
| $D_6$      | Chicago      | TBB     | 2024-10 | 100 × 100  |
| $D_7$      | Chicago      | Chrome  | 2024-7  | 100 × 100  |
| $D_8$      | Chicago      | TBB     | 2024-7  | 4000 × 1   |


## Public Dataset
We sincerely thank the authors for sharing their dataset. Two public real-world datasets used in our experiments are listed as follows:



* [Wang dataset](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/wang_tao): It contains 100 websites, each with 90 undefended traces, for closed-world evaluation. Excluding the 100 websites, it also includes 9,000 websites for open-world evaluation, each with only 1 undefended trace.
This dataset is provided by [Tao Wang et al.](https://www.cs.sfu.ca/~taowang/wf/index.html), and you can find the dataset on this [link](https://www.cs.sfu.ca/~taowang/wf/index.html).

* [DF dataset](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768): It contains 95 websites, each with 1,000 undefended traces, for closed-world evaluation. Excluding the 95 websites, it also includes 40,000 websites for open-world evaluation, each with only 1 undefended trace. 
This dataset is provided by [Rahman et al.](https://github.com/msrocean/Tik_Tok), and you can find the dataset on the [google drive link](https://drive.google.com/file/d/1q0wWJPEtaXmv53QIT5tErxVXv3QokwUg/view?usp=drive_link).




# The code of attacks and defenses

We sincerely thank all the researchers for providing their code.

## Attacks
| Attacks | Conference    | Paper                                                        | Code                                                                                                  |
|---------|---------------| ------------------------------------------------------------ |-------------------------------------------------------------------------------------------------------|
| DF      | CCS 2018      | [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768) | [DF](https://github.com/deep-fingerprinting/df)                                                       |
| Tik-Tok | PETS 2019     | [Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks](https://petsymposium.org/popets/2020/popets-2020-0043.pdf) | [Tik-Tok](https://github.com/msrocean/Tik_Tok)                                                        |
| Var-CNN | PETS 2019     | [Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning](https://arxiv.org/pdf/1802.10215) | [Var-CNN](https://github.com/sanjit-bhat/Var-CNN)                                                     |
| TF      | CCS 2019      | [Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning](https://dl.acm.org/doi/pdf/10.1145/3319535.3354217) | [TF](https://github.com/triplet-fingerprinting/tf)                                                    |
| RF      | Security 2023 | [Subverting Website Fingerprinting Defenses with Robust Traffic Representation](https://www.usenix.org/system/files/sec23fall-prepub-621_shen-meng.pdf) | [RF](https://github.com/robust-fingerprinting/RF)                                                     |
| NetCLR  | CCS 2023      | [Realistic Website Fingerprinting By Augmenting Network Trace](https://arxiv.org/pdf/2309.10147) | [NetCLR](https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces) |
| Swallow | CCS 2025      | [Swallow: A Transfer-Robust Website Fingerprinting Attack via Consistent Feature Learning](https://arxiv.org/pdf/2407.00918) | [Swallow](https://github.com/wujinhe0814/Swallow)                                                     |

## Defenses 

| Defenses      | Conference    | Paper                                                                                    | Code                                                                      |
|---------------|---------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Tamaraw       | CCS 2014      | [A systematic approach to developing and evaluating website fingerprinting defenses](https://dl.acm.org/doi/abs/10.1145/2660267.2660362 ) | [Tamaraw](https://www.cs.sfu.ca/~taowang/wf/index.html)                   |
| WTF-PAD       | ESORICS 2016  | [Toward an efficient website fingerprinting defense](https://link.springer.com/chapter/10.1007/978-3-319-45744-4_2)          | [WTF-PAD](https://github.com/websitefingerprinting/WebsiteFingerprinting/) |
| Front         | Security 2020 | [Zero-delay lightweight defenses against website fingerprinting](https://www.usenix.org/conference/usenixsecurity20/presentation/gong) | [Front](https://github.com/websitefingerprinting/WebsiteFingerprinting/)  |
| TrafficSliver | CCS 2020      | [Trafficsliver: Fighting website fingerprinting attacks with traffic splitting](https://dl.acm.org/doi/abs/10.1145/3372297.3423351)       | [TrafficSliver](https://github.com/TrafficSliver/splitting_simulator)     |
| RegulaTor     | PETS 2022     | [RegulaTor: A straightforward website fingerprinting defense](https://arxiv.org/abs/2012.06609)  | [RegulaTor](https://github.com/jkhollandjr/RegulaTor)                     |
| Surakav       | S&P 2022      | [Surakav: Generating realistic traces for a strong website fingerprinting defense](https://www.example.com ) | [Surakav](https://github.com/websitefingerprinting/surakav-imp)           |
| Palette       | S&P 2024      | [Real-Time Website Fingerprinting Defense via Traffic Cluster Anonymization](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a263/1WPcZnZILHa) | [Palette](https://github.com/kxdkxd/Palette)            |


# Contact
If you have any questions, please get in touch with us.

* Prof. Meng Shen ([shenmeng@bit.edu.cn](shenmeng@bit.edu.cn))
* Jinhe Wu ([jinhewu@bit.edu.cn](jinhewu@bit.edu.cn))
* Junyu Ai ([aijunyu@bit.edu.cn](aijunyu@bit.edu.cn))

More detailed information about the research of Meng Shen Lab can be found [here](https://mengshen-office.github.io/).