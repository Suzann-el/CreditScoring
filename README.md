# 💳 Prêt à Dépenser — Credit Scoring

> Modèle de scoring crédit avec dashboard interactif pour l'aide à la décision d'octroi de prêt.

Lien vers l'API Render : https://scoring-1-2ylz.onrender.com/docs
---

## 🎯 Contexte

Dans le secteur du crédit à la consommation, évaluer le risque de défaut de paiement est critique — surtout pour des clients avec peu ou pas d'historique bancaire. Ce projet propose une solution end-to-end : de la modélisation ML jusqu'au déploiement d'un dashboard métier.

---

## ⚙️ Ce que fait le projet

- **Modèle de scoring** — prédit la probabilité de défaut de remboursement d'un client
- **Seuil métier optimisé** — minimise le coût asymétrique entre faux positifs et faux négatifs
- **API FastAPI** — expose les prédictions en temps réel
- **Dashboard Streamlit** — permet aux chargés de relation client de visualiser et d'expliquer les décisions

## 📊 Fonctionnalités du dashboard

- Score client avec jauge visuelle et interprétation en langage naturel
- Explication locale des décisions via **SHAP**
- Comparaison du profil client vs la population globale ou un groupe similaire
- Exploration des variables descriptives par filtres

---

## 🛠️ Stack

| Couche | Outils |
|--------|--------|
| Modélisation | `LightGBM` `Scikit-learn` `imbalanced-learn` |
| Explicabilité | `SHAP` |
| API | `FastAPI` `Uvicorn` |
| Dashboard | `Streamlit` `Plotly` |
| Déploiement | `Render` |
| Suivi | `MLflow` |

---

## 📁 Structure du projet

```
├── data/               # Données brutes et features engineerées
├── notebooks/          # Exploration, feature engineering, modélisation
├── api/                # Application FastAPI
├── dashboard/          # Application Streamlit
└── models/             # Modèles sérialisés
```

---

## 🚀 Lancer le projet en local

```bash
# Cloner le repo
git clone https://github.com/Suzann-el/Scoring.git

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api.main:app --reload

# Lancer le dashboard
streamlit run dashboard/app.py
```

---

## 📂 Données

Données issues du concours Kaggle **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)** — fichiers clients, historique de crédits, données comportementales.

---

## 🔗 Démo

👉 [Dashboard en ligne](https://dashboard-client-app.herokuapp.com/)
