# PhiSN: Phishing URL Detection using Segmentation and NLP Features
This is an implementation of my paper "PhiSN: Phishing URL Detection using Segmentation and NLP Features."

Published in Journal of Information Processing, 2024, Volume 32, Pages 973-989.
Released on J-STAGE November 15, 2024, Online ISSN 1882-6652.
Available @.https://doi.org/10.2197/ipsjjip.32.973
### Cite https://doi.org/10.2197/ipsjjip.32.973

## Abstract
Phishing is a cyberattack method employed by malicious actors who impersonate authorized personnel to illicitly obtain confidential information. Phishing URLs constitute a form of URL-based intrusion designed to entice users into disclosing sensitive data. This study focuses on URL-based phishing detection, which involves identifying phishing webpages by analyzing their URLs. We primarily address two unresolved issues of previous research: (i) insufficient word tokenization of URLs, which leads to the inclusion of meaningless and unknown words, thereby diminishing the accuracy of phishing detection, (ii) dataset-dependent lexical features, such as phish-hinted words extracted exclusively from a given dataset, which restricts the robustness of the detection method. To solve the first issue, we propose a new segmentation-based word-level tokenization algorithm called SegURLizer. The second issue is addressed by dataset-independent natural language processing (NLP) features, incorporating various features extracted from domain-part and path-part of URLs. Then, we employ the segmentation-based features with SegURLizer on two neural network models - long short-term memory (LSTM) and bidirectional long short-term memory (BiLSTM). We then train the 36 selected NLP-based features on deep neural networks (DNNs). Afterward, we combine the outputs of the two models to develop a hybrid DNN model, named “PhiSN, ” representing phishing URL detection through segmentation and NLP features. Our experiments, conducted on five open datasets, confirmed the higher performance of our PhiSN model compared to state-of-the-art methods. Specifically, our model achieved higher F1-measure and accuracy, ranging from 95.30% to 99.75% F1-measure and from 95.41% to 99.76% accuracy, on each dataset.

### Steps:
0: Install necessary packages mentioned in "installation_package_version.txt"
1: Segment URLs using SegURLizer algorithm
2: Cross-validate data
3: Trian and Test Hybrid model, i.e., connected from Segmentation-based BiLSTM and NLP-based DNN
4: Trian and Test Hybrid model, i.e., connected from Segmentation-based LSTM and NLP-based DNN