# PS-SVM: Probabilistic Slack SVM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onetwomanye/ps-svm/blob/main/PS_SVM_Experiments.ipynb)

sklearn-compatible PS-SVM with principled uncertainty quantification.

```python
from ps_svm import PSSVMClassifier
ps = PSSVMClassifier(C=1.0, loss='l2')   # or loss='huber'
ps.fit(X_train, y_train)
proba = ps.predict_proba(X_test)
print(ps.ergodicity_report())  # Q_fraction + four ergodicity measures
```

*NTU / Freeboh Innovations 2026*
