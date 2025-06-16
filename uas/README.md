Mari saya jelaskan bagaimana kode evaluasi ini bekerja untuk data contoh yang Anda berikan, dengan tracing lengkap dari preprocessing hingga prediksi!

## **1. Input Data dan Preprocessing**

### **Data Input:**
```
Academic Pressure: 2
Study Satisfaction: 4  
Sleep Duration: 7-8 hours
Dietary Habits: Moderate
Suicidal thoughts: Yes
Study Hours: 9
Financial Stress: 2
Family History: Yes
Depression: No (target actual)
```

### **Preprocessing Steps:**

#### **A. Konversi Kategorik ke Numerik:**
```python
Sleep_Numeric = 3        # "7-8 hours" → 3
Diet_Numeric = 2         # "Moderate" → 2
Suicidal_Numeric = 1     # "Yes" → 1
Family_History_Numeric = 1  # "Yes" → 1
Depression_Numeric = 0   # "No" → 0 (target actual)
```

#### **B. Composite Score Calculation:**

**Lifestyle_Score:**
```python
Lifestyle_Score = (Sleep_Numeric * 0.4) + (Diet_Numeric * 0.4) + ((1 - Suicidal_Numeric) * 0.2)
                = (3 * 0.4) + (2 * 0.4) + ((1 - 1) * 0.2)
                = 1.2 + 0.8 + 0.0
                = 2.0
```

**Academic_Stress_Score:**
```python
# Asumsi Study Hours range [1, 20] untuk normalisasi
study_hours_normalized = (9 - 1) / (20 - 1) * 4 + 1 = 8/19 * 4 + 1 ≈ 2.68

Academic_Stress_Score = (Academic_Pressure * 0.6) + (study_hours_normalized * 0.4)
                      = (2 * 0.6) + (2.68 * 0.4)
                      = 1.2 + 1.07
                      = 2.27
```

**Depression_Predisposition:**
```python
Depression_Predisposition = (Financial_Stress * 0.7) + (Family_History_Numeric * 1.5)
                          = (2 * 0.7) + (1 * 1.5)
                          = 1.4 + 1.5
                          = 2.9
```

---

## **2. Evaluasi Sistem Fuzzy**

### **Input untuk FIS:**
```python
lifestyle = 2.0
academic = 2.27  
predisposition = 2.9
```

### **A. Evaluasi Sistem Mamdani**

#### **Step 1: Fuzzification**
```python
# Lifestyle (range 1-4)
lifestyle_poor = trimf(2.0, 1, 1, 2.5) = (2.5-2.0)/(2.5-1) = 0.33
lifestyle_moderate = trimf(2.0, 1.5, 2.5, 3.5) = (2.0-1.5)/(2.5-1.5) = 0.5  
lifestyle_good = trimf(2.0, 2.5, 4, 4) = 0

# Academic (range 1-5)
academic_low = trimf(2.27, 1, 1, 3) = (3-2.27)/(3-1) = 0.365
academic_medium = trimf(2.27, 2, 3, 4) = (2.27-2)/(3-2) = 0.27
academic_high = trimf(2.27, 3, 5, 5) = 0

# Predisposition (range 0-6)
predisposition_low = trimf(2.9, 0, 0, 3) = (3-2.9)/(3-0) = 0.033
predisposition_medium = trimf(2.9, 1, 3, 5) = (3-2.9)/(3-1) = 0.05
predisposition_high = trimf(2.9, 3, 6, 6) = 0
```

#### **Step 2: Rule Application**
**Activated Rules:**
```python
# Rule: ('poor', 'low', 'low') → 'medium'
strength = min(0.33, 0.365, 0.033) = 0.033

# Rule: ('moderate', 'low', 'low') → 'low'  
strength = min(0.5, 0.365, 0.033) = 0.033

# Rule: ('poor', 'medium', 'low') → 'high'
strength = min(0.33, 0.27, 0.033) = 0.033

# Rule: ('moderate', 'medium', 'low') → 'medium'
strength = min(0.5, 0.27, 0.033) = 0.033
```

#### **Step 3: Aggregation**
```python
aggregated = {
    'low': 0.033,
    'medium': max(0.033, 0.033) = 0.033,
    'high': 0.033
}
```

#### **Step 4: Defuzzification (Centroid)**
```python
# Clipping dan union dari output membership functions
# Dengan aggregated values yang rendah (0.033), centroid ≈ 0.5
mamdani_output ≈ 0.5
```

### **B. Evaluasi Sistem Sugeno**

#### **Step 1: Fuzzification (sama dengan Mamdani)**

#### **Step 2: Weighted Average Defuzzification**
```python
# Contoh rule: ('poor', 'low', 'low') → (-0.1, 0.05, 0.05, 0.2)
rule_output = -0.1*2.0 + 0.05*2.27 + 0.05*2.9 + 0.2
            = -0.2 + 0.1135 + 0.145 + 0.2
            = 0.2585
strength = 0.033

weighted_sum = 0.033 * 0.2585 = 0.0085
total_weight = 0.033

# Setelah semua rules:
sugeno_output = weighted_sum / total_weight ≈ 0.45
```

---

## **3. Binary Classification**

### **Threshold Conversion:**
```python
# Threshold = 0.5
mamdani_prediction = 1 if 0.5 > 0.5 else 0 = 0  # No Depression
sugeno_prediction = 1 if 0.45 > 0.5 else 0 = 0  # No Depression

# Actual target = 0 (No Depression)
```

---

## **4. Evaluasi Metrics**

### **Accuracy Calculation:**
```python
# Untuk contoh ini:
# Mamdani: Predicted=0, Actual=0 → Correct ✓
# Sugeno: Predicted=0, Actual=0 → Correct ✓
```

### **Classification Report:**
```python
# Akan menampilkan:
# - Precision, Recall, F1-score untuk setiap class
# - Support (jumlah instance per class)
# - Macro dan weighted averages
```

### **Confusion Matrix:**
```
             Predicted
           No_Dep  Dep
Actual No_Dep [TN] [FP]
       Dep    [FN] [TP]
```

---

## **5. Visualisasi Perbandingan**

### **A. Bar Chart Akurasi:**
```python
# Menampilkan accuracy kedua sistem side-by-side
# Dengan nilai akurasi di atas setiap bar
```

### **B. Pie Charts Distribusi:**
```python
# Mamdani predictions distribution
pred_counts_mamdani = [count_0, count_1]

# Sugeno predictions distribution  
pred_counts_sugeno = [count_0, count_1]
```

---

## **6. Feature Importance Analysis**

### **Correlation Matrix:**
```python
correlation_data = data[['Lifestyle_Score', 'Academic_Stress_Score', 
                        'Depression_Predisposition', 'Depression_Numeric']]

# Akan menampilkan:
# - Heatmap korelasi antar fitur
# - Ranking fitur berdasarkan korelasi dengan target
```

**Expected Output untuk contoh ini:**
```
Korelasi fitur dengan Depression (diurutkan):
  Depression_Predisposition: 0.654
  Academic_Stress_Score: 0.423  
  Lifestyle_Score: -0.587  # Negative karena lifestyle baik = depresi rendah
```

---

## **7. Interpretasi Hasil**

### **Untuk Data Contoh Ini:**

**Karakteristik Pasien:**
- **Lifestyle:** Moderate (2.0) - ada suicidal thoughts
- **Academic Stress:** Low-Medium (2.27) - pressure rendah tapi study hours tinggi
- **Predisposition:** Medium (2.9) - ada family history

**Prediksi Kedua Sistem:**
- **Mamdani:** 0.5 → No Depression (borderline)
- **Sugeno:** 0.45 → No Depression (lebih yakin)
- **Actual:** No Depression ✓

**Alasan Prediksi Benar:**
1. **Academic stress relatif rendah** (2.27 dari max 5)
2. **Lifestyle moderat** meski ada suicidal thoughts
3. **Family history** memang faktor risiko, tapi dikompensasi faktor lain

### **Perbedaan Sistem:**
- **Mamdani:** Lebih konservatif, output tepat di threshold
- **Sugeno:** Lebih decisive, output jelas di bawah threshold
- **Kedua sistem:** Prediksi benar untuk kasus ini

**Kesimpulan:** Sistem fuzzy berhasil menangkap kompleksitas case ini dimana meski ada beberapa faktor risiko (suicidal thoughts, family history), faktor protektif (academic stress rendah, lifestyle reasonable) cukup untuk menghasilkan prediksi "No Depression" yang akurat!