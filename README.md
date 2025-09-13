# Türkçe Haber Sınıflandırma

TF-IDF → SVD (300) → **Train'de SMOTE (min %5)** → LinearSVC/LogReg seçimi (Val macro-F1) → Test değerlendirme.

## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
