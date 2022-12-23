# PWWS Replication
Our group project replicates the PWWS proposed in [Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency](https://aclanthology.org/P19-1103) (Ren et al., ACL 2019).
## Dataset
We use three datasets: IMDB Review, AG's News, and Yahoo! Answers. You can download from [Google Drive](https://drive.google.com/drive/folders/17uMfWw422w2MekjztLqK1htN-DKO2eVm?usp=share_link)
The preprocess steps are included in *data_reader.py* and *process.py*
## Models
We train three models as classifiers: word-based CNN, LSTM, and bi-directional LSTM. The model structures are defined in *models.py*. The training details are in *config.py*. We use *train.py* to train these models.
## Attack
We use *pwws_attacker.py* to generate adversarial examples with several helper functions in *utils.py* and *word_saliency.py*.
## Evaluation
We use *evaluate.py* to check the model performance and *evaluate_pwws.py* to validate the functionality of PWWS.

