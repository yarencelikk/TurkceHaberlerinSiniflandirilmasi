# Türkçe Haber Sınıflandırma (ML Pipeline)

Bu proje, Türkçe haber metinlerini çok sınıflı olarak sınıflandırır. Boru hattı:

**Temizleme → TF-IDF → SVD(300) → (yalnızca train’de) SMOTE(min %5) → Model seçimi (Val macro-F1) → Test değerlendirme.**

> Not: SMOTE **sadece eğitim setine** uygulanır; doğrulama/test doğal dağılımda kalır (adil değerlendirme).

---

## 📁 Depo İçeriği

- `haber_siniflandirmaSon.ipynb` — Eğitim/EDA/Split/SMOTE/Model kaydetme
- `Haber_tahmin.ipynb` — Tek hücrelik, **meta uyumlu** tahmin aracı
- `news.csv` — Örnek veri (istersenizi repoda tutmayın; alternatifi aşağıda)

Eğitim defteri çalışınca şu dosyalar oluşur:
- `best_meta.json` — Model/vektör/SVD dosya adları + sınıf listesi
- `best_vectorizer.joblib` — TF-IDF vektörleyici
- `best_svd.joblib` — TruncatedSVD(300)
- `<seçilen_model>.joblib` — (örn. `LinearSVC_bal.joblib`)

---

## 🚀 Hızlı Kurulum 

Python 3.10+ önerilir.

```bash
pip install pandas numpy scikit-learn>=1.2 imbalanced-learn>=0.12 \
matplotlib seaborn scipy joblib nltk TurkishStemmer
