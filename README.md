# calling-out-bluff

Public Implementation of paper Calling Out Bluff: Attacking the Robustness of Automatic Scoring Systems with Simple Adversarial Testing. </br>
[Arxiv Link](http://arxiv.org/abs/2007.06796) </br>
Authors:
Yaman Kumar*, Mehar Bhatia*, Anubha Kabra*, Jessy Junyi Li, Di Jin, Rajiv Ratn Shah
</br>
For any questions or issues, feel free to email us at [yamank@iiitd.ac.in](mailto:yamank@iiitd.ac.in), [mehar.bhatia@midas.center](mailto:mehar.bhatia@midas.center).

### Generating Adversarial Samples ###
To generate our adversarial samples, please view the file 'TestCaseSuite_CallingOutBluff.ipynb'
Working of this TestCaseSuite file :
1. Download ASAP AES Dataset from [here.](https://www.kaggle.com/c/asap-aes)
2. Save the training and test cases prompt wise.
3. Change the prompt number and file to load in the first cell of the notebook.
4. Download and load the supporting files for test cases, given below.
5. The notebook has been commented well, find the specifc test case you want to simulate and run!

### Our Adversarial testcases can be found here:
To view all our simulated adversarial testcases, click below
1. [ASAP-AES dataset](https://drive.google.com/open?id=1CIEpiDmzLmJ6LMCVSOmCKw_eOg4ocuS4)
2. [ASAP-SAS dataset](https://drive.google.com/drive/folders/1oWP31zo02009skA24nC10tYlCGWqOAOx)

### Some supporting files:
Sentence list for out testcases like songs, speech, wikipedia, universal false, universal truths etc can be found [here](https://drive.google.com/open?id=1hYQ-GtuQVcMYIeGcvBCTB6wXUHxOC1aY)

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
  * Folder: Model4(BERT)

5. MemoryNetworks
  * [Paper Link](https://par.nsf.gov/servlets/purl/10060135)
  * Folder: Model5(MemoryNets)
