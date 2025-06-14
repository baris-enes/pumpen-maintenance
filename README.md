# Predictive Maintenance für Industrie­pumpen

Dieses Projekt demonstriert einen vollständigen Machine-Learning-Workflow zur prädiktiven Wartung von Industrie­pumpen – mit besonderem Fokus auf die **chemische Industrie**. Durch gezieltes Feature Engineering und datengetriebenes Clustering entsteht ein belastbares Simulationsmodell zur frühzeitigen Erkennung von Wartungsbedarf.

---

##  Datengrundlage

Die zugrundeliegenden Daten stammen aus einem synthetischen Pumpenbetriebsszenario. Jeder Datenpunkt beschreibt den Zustand einer Pumpe mit Messgrößen wie:

- **Temperature** (°C)
- **Vibration** (mm/s)
- **Pressure** (bar)
- **Flow Rate** (m³/h)
- **RPM** (Drehzahl)
- **Operational Hours** (h)

Zusätzlich enthält der Datensatz die Zielvariable `Maintenance_Flag` (0 = kein Wartungsbedarf, 1 = Wartung notwendig). Allerdings zeigte sich, dass dieses Label **kaum trennscharf** war (Verhältnis nahezu 50/50, kaum Korrelation zu Features). Daher wurde im Projektverlauf eine **simulierte Zielvariable** eingeführt, um ein realistisches und erklärbares Wartungsmodell zu ermöglichen.

---

##  Projektübersicht

1. **Datenexploration & Ausgangssituation**
2. **Physikalisch motiviertes Feature Engineering**
3. **Clustering & Label-Simulation**
4. **Modelltraining & Hyperparameter-Optimierung (Grid Search)**
5. **Evaluation mit Konfusionsmatrix**

---

##  Verwendete Tools & Bibliotheken

- **Python 3.8+**
- **pandas** – Datenmanipulation & Einlesen (`pd.read_csv`)
- **NumPy** – numerische Berechnungen (`np.log1p`, Arithmetik)
- **scikit-learn**
  - `StandardScaler`, `PCA`, `KMeans` (Clustering)
  - `train_test_split`, `GridSearchCV` (Modell-Validierung)
  - `classification_report`, `confusion_matrix` (Metriken)
- **XGBoost** (`xgboost.XGBClassifier`) – Gradient Boosted Trees
- **matplotlib** & **seaborn** – Visualisierung (Heatmaps, Scatter, Bar Charts, Confusion Matrix)

> **Hinweis:** TensorFlow oder andere Deep-Learning-Frameworks kamen **nicht** zum Einsatz – der Fokus liegt auf klassischer ML-Pipeline.

---

##  Installation

```bash

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

*requirements.txt* sollte enthalten:

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

---

##  Methodik

### Feature Engineering

Gezielt eingeführte Features mit physikalischem Bezug:

| Feature                  | Formel                           | Bedeutung                            |
|--------------------------|----------------------------------|--------------------------------------|
| Power                    | Pressure × Flow Rate             | Hydraulische Leistung                |
| Vibration_per_hour       | Vibration / Operational_Hours    | Belastungsrate                       |
| Efficiency               | Power / RPM                      | Betriebseffizienz                    |
| Cumulative_Load          | Vibration × Operational_Hours    | Gesamtbelastung über Zeit           |
| Log_Vibration_per_hour   | log(1 + Vibration_per_hour)      | Skalierung zur Varianzreduktion     |


Diese Features führten zu einer **deutlich erhöhten Korrelation mit dem Wartungsbedarf** (sichtbar in Korrelationsmatrizen vor/nach Feature Engineering).

### Clustering & Label-Simulation

Da das ursprüngliche Label unbrauchbar war, wurde ein **domänengetriebenes Pseudo-Labeling** durchgeführt:

- **PCA → KMeans mit 3 Clustern**
- Neue Zielvariable: `Maintenance_Flag_Sim_Final`
  - **Cluster 2** = immer Wartung notwendig
  - **Cluster 0** = nie Wartung
  - **Cluster 1** = Regelbasiert:
    - Hohe kumulative Belastung & niedrige Effizienz ⇒ Wartung

### Modelltraining & Grid Search

- Modell: `XGBClassifier`
- Ziel: Optimierung auf **F1-Score**
- Beste Parameter laut GridSearch:
  ```
  n_estimators:     200  
  max_depth:        3  
  learning_rate:    0.10  
  subsample:        0.80  
  colsample_bytree: 0.80
  ```

### Evaluation

Die finale Konfusionsmatrix zeigt eine **nahezu perfekte Klassifikation** auf Basis der simulierten Labels.

---

##  Relevanz für die chemische Industrie

In der chemischen Industrie sind Pumpen zentrale Aggregate zur Förderung von Medien wie:

- **Säuren, Laugen, Lösungsmittel, Emulsionen**
- Stoffe mit **hoher Viskosität** oder **abrasiven Eigenschaften**
- Medien mit **kritischen Temperatur- und Druckanforderungen**

Daher sind vorausschauende Wartungsstrategien essenziell. Dieses Projekt zeigt, wie:

- **Prozessdaten + domänenspezifisches Wissen**
- zu **robusten Features** führen
- und daraus **frühzeitige Wartungsempfehlungen** generiert werden können.

---

##  Lessons Learned

-  **Physik schlägt Zufall:** Nur durch Feature Engineering entstand echte Korrelation
-  **Domain-Knowledge + Clustering** helfen bei der Labelkonstruktion
-  **ML-Modelle sind nur so gut wie ihre Eingangsdaten** – Featurequalität entscheidet

---

##  Nächste Schritte

- Einbindung von **Medienkennwerten** wie Dichte, Korrosivität, chemischer Zusammensetzung
- - Erweiterung um **Zeitreihenanalysen mit TensorFlow**:
  - Gleitende Mittelwerte (Rolling Features)
  - Differenzen (z. B. ΔVibration)
  - Trend-Erkennung mit LSTM oder TCN-Modellen
  - Ziel: Frühzeitige Detektion schleichender Defekte
- Umsetzung als **API-Service** (z. B. FastAPI + Docker)
- Integration in Dashboard (z. B. Streamlit, Dash) für Produktionsleiter

---

*© 2025 Enes Baris – basierend auf einem synthetischen Pumpendatensatz*

