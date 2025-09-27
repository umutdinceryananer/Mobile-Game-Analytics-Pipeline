
## Proje Amacı

Mobil oyun kullanıcı davranışlarını üretilen veriseti üzerinden analiz ederek iş ilanında geçen veri bilimi beklentilerini karşılamak:

* Kullanıcı dönüşüm hunisi (funnel) analizi
* ROI ve pazarlama performansı değerlendirmesi
* Retention ve churn analizi
* Gelir tahmini (forecasting)
* SQL, Python ve dashboard kullanımı

---

## Zaman Planı (3 Günlük Çalışma)

### Gün 1 — Veri Keşfi & Hazırlık

* [ ] Kaggle’dan **Cookie Cats A/B Test Dataset** indir
* [ ] Sentetik veri ekleme (`acquisition_channel`, `CAC`, `revenue`, `ROI`)
* [ ] Data exploration: kullanıcı sayısı, seans dağılımı, satın alma oranı
* [ ] Eksik veri, outlier, duplicate kontrolü
* [ ] SQL ile basit sorgular (ör. günlük install sayısı)

**Input:** raw_data.csv
**Output:** /processed/clean_data.csv, parquet file, eda.ipynb çıktıları, başarılı pytest sonuçları

**Başarı Kriterleri:**

* [ ] Eksik veriler %5’in altında
* [ ] Sentetik değişkenler başarılı şekilde eklendi
* [ ] İlk görselleştirmeler (distribution plots) üretildi

---

### Gün 2 — Funnel & ROI Analizi

* [ ] Funnel dönüşüm analizi (ad click → install → session → purchase)
* [ ] Cohort retention (D1, D7, D30)
* [ ] ROI hesaplama (sahte CPI eklenerek)
* [ ] Görselleştirmeler: matplotlib/plotly + Tableau

**Input:** /processed/clean_data.csv
**Output:** sql_analysis.ipynb, funnel grafikleri

**Başarı Kriterleri:**

* [ ] Funnel her aşama için conversion % gösteriyor
* [ ] Cohort retention tablosu hazırlandı
* [ ] ROI kanal bazlı hesaplandı

---

### Gün 3 — Prediction, Forecasting & Dashboard

* [ ] Churn label tanımlama (ör. 7 gün aktif olmayan kullanıcı = churn)
* [ ] Churn prediction modeli (Logistic Regression, XGBoost)
* [ ] Revenue forecasting (Prophet veya ARIMA)
* [ ] Tableau dashboard → Funnel, retention, churn risk, forecast
* [ ] Executive Summary raporu (PDF/Markdown)

**Input:** /processed/clean_data.csv
**Output:** modeling.ipynb, dashboard screenshotları, executive_summary.md

**Başarı Kriterleri:**

* [ ] Churn modeli AUC ≥ 0.75
* [ ] Forecast grafiklerinde trend + sezon etkisi görülebiliyor
* [ ] Dashboard’da en az 4 ana grafik var

---

## İş Odaklı Kavramlar

* **Conversion Funnel** → Kullanıcı yolculuğu (ad click → install → session → purchase)
* **Retention (D1, D7, D30)** → Kullanıcıların oyunda kalma oranı
* **Churn** → Oyunu bırakan kullanıcı oranı
* **Cohort Analysis** → Kayıt tarihine göre retention/LTV farklılıkları
* **ARPU (Average Revenue Per User)** → Kullanıcı başına ortalama gelir
* **ARPPU** → Ödeme yapan kullanıcı başına ortalama gelir
* **LTV (Lifetime Value)** → Kullanıcının yaşam boyu ürettiği gelir
* **CPI (Cost Per Install)** → Kullanıcı kazanma maliyeti
* **ROI (Return on Investment)** → (LTV – CPI) / CPI

---

## Veri Sözlüğü (Data Dictionary)

| Sütun Adı           | Açıklama                                   | Tipi        |
| ------------------- | ------------------------------------------ | ----------- |
| user_id             | Kullanıcı kimliği                          | String      |
| install_date        | Oyunun kurulduğu tarih                     | Date        |
| acquisition_channel | Kullanıcıyı getiren kanal (FB, Google vb.) | Categorical |
| CAC                 | Kullanıcı başına edinim maliyeti           | Numeric     |
| revenue             | Kullanıcının getirdiği toplam gelir        | Numeric     |
| sessions            | Toplam oturum sayısı                       | Integer     |
| retention_d1/d7/d30 | Retention flag’leri                        | Boolean     |
| churn_flag          | Kullanıcının churn olup olmadığı           | Boolean     |

---

## Executive Summary (Örnek)

* Funnel dönüşüm oranı: %12 kullanıcı install → purchase
* D7 retention: %23 → churn oranı yüksek
* Churn model AUC: 0.78 → düşük session count churn riski ile ilişkili
* Revenue forecast: 30 gün içinde gelir trendinde %5 artış bekleniyor

---

## Sonraki Geliştirme Önerileri

* Dashboard’a kullanıcı segmentasyonu eklenmesi (ör. ülke/cihaz bazlı)
* Farklı ML algoritmalarıyla churn modeli kıyaslama
* Gerçekçi synthetic revenue dağılımı için parametre tuning
* Kullanıcı yaşam boyu değeri (LTV) için survival analysis


### Changelog
- 2025-09-27: data/make_dataset.py dosyasına ülke ve platform sentetik kolonları eklendi, platform bazlı gelir ölçekleme desteği sağlandı.

- 2025-09-27: data/config/synthetic.yaml dosyasında geo.country ve platform yapılandırmaları tanımlandı.
