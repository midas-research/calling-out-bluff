# calling-out-bluff

Public Implementation of paper Calling Out Bluff: Attacking the Robustness of Automatic Scoring Systems with Simple Adversarial Testing.
[Arxiv Link](http://arxiv.org/abs/2007.06796)
Authors:
Yaman Kumar*, Mehar Bhatia*, Anubha Kabra, Jessy Junyi Li, Di Jin, Rajiv Ratn Shah

### Generating testcases ###
To generate our testcases, please view the file 'TestCaseSuite_CallingOutBluff.ipynb'

### Our testcases can be found here:
1. [ASAP-AES dataset](https://drive.google.com/open?id=1CIEpiDmzLmJ6LMCVSOmCKw_eOg4ocuS4)
2. [ASAP-SAS dataset](https://drive.google.com/drive/folders/1oWP31zo02009skA24nC10tYlCGWqOAOx)

Sentence list for out testcases like songs, speech, wiki, universal false, universal truths etc can be found [here](https://drive.google.com/open?id=1hYQ-GtuQVcMYIeGcvBCTB6wXUHxOC1aY)

### Models: 

1. LSTM with MoT layer 
  * [Paper Link](https://www.aclweb.org/anthology/D16-1193/)
  * Folder: Model1(LSTM_MoT)
  * Weights uploaded


2. EASE
  * [Implementation Link](https://github.com/edx/ease/)
  * Folder: Model2(EASE)

3. Skipflow
  * [Paper Link](https://arxiv.org/abs/1711.04981)
  * Folder: Model3(SkipFlow)
  * Please download glove.6B.300d embeddings and save in main folder. (Too large to add)
  * Weights uploaded

4. BERT+Adversarial Evaluation (Two Stage Learning)
  * [Paper Link](https://arxiv.org/abs/1901.07744)
  * Model4(BERT)

5. MemoryNetworks
  * [Paper Link](https://par.nsf.gov/servlets/purl/10060135)
  * Model5(MemoryNets)
