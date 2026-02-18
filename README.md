# tfa_examples
Examples with TensorFlowAnalysis2

To install:
 - clone [AmpliTF](https://github.com/cpviolation/AmpliTF) and [TFA2](https://github.com/cpviolation/TFA2) in the same directory as `tfa_examples`
   
   **n.b.**: it is advisable to fork the repositories to your github account and define the following remotes
   for AmpliTF
   ```bash
   git remote add cpviolation git@github.com:cpviolation/AmpliTF.git
   git remote add upstream git@github.com:apoluekt/AmpliTF.git
   ```
   and for TFA2
   ```bash
   git remote add cpviolation git@github.com:cpviolation/TFA2.git
   git remote add upstream git@github.com:apoluekt/TFA2.git
   ```
 - prepare a python environment
   ```bash
   python3 -m venv tfa
   source tfa/bin/activate
   ```
 - install the requirements (`requirements_mac.txt` if you are installing on Apple Silicon)
   ```bash
   pip install -r requirements.txt
   ```
 - call `setup.sh` after loading the environment to define variables 
