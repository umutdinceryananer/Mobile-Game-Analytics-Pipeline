
## Proje Amacı

Mobil oyun kullanıcı davranışlarını üretilen veriseti üzerinden analiz ederek iş ilanında geçen veri bilimi beklentilerini karşılamak:

* Kullanıcı dönüşüm hunisi (funnel) analizi
* ROI ve pazarlama performansı değerlendirmesi
* Retention ve churn analizi
* Gelir tahmini (forecasting)
* SQL, Python ve dashboard kullanımı

---

## Zaman Planı (3 Günlük Çalışma)

### Day 1 — Data Exploration & Preparation (completed)

Processed the Cookie Cats dataset, verified no missing values or duplicate user IDs.
Added synthetic columns deterministically, driven by configuration:
* [ ] acquisition_channel (Instagram, Facebook, TikTok, Organic)
* [ ] CAC (channel-based customer acquisition cost)
* [ ] purchase (session-based probability)
* [ ] revenue (in-app purchases via gamma distribution + ad revenue per session)
* [ ] ROI (return on investment per user)
* [ ] country (USA, Mexico, Brazil, with realistic distribution)
* [ ] platform (App Store, Google Play, calibrated to installs and revenue split)

Updated synthetic.yaml: increased CAC variance, scaled up revenue parameters, calibrated platform revenue shares to match 26M vs 9M distribution.
Added automated tests (pytest, 6 tests) to validate schema, distributions, and revenue shares. All tests passed.

Initial EDA results:
* [ ] 90,189 rows, 12 columns
* [ ] Global purchase rate: 5.58%
* [ ] Retention: D1 ≈ 44.5%, D7 ≈ 18.6%
* [ ] Average sessions: 51.9 (median 16, heavy-tailed, >425 sessions flagged as outliers)
* [ ] Channel, platform, and country distributions match configuration
* [ ] Platform performance: App Store ≈ 24.9% of installs but ≈ 74.4% of revenue; Google Play ≈ 75.1% of installs and ≈ 25.6% of revenue
* [ ] Revenue and ROI distributions widened; ROI remains negative on average but is suitable for scenario analysis

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
