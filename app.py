# -*- coding: utf-8 -*-
"""
🛡️ DDoS Shield — Détection et prédiction d'attaques DDoS
Application Streamlit basée sur le pipeline ML de l'atelier (YaneCodeAcademy).

Lancement :  streamlit run app.py
Identifiants : testDDoS / havefun

NB : le dataset est SYNTHÉTIQUE (usage pédagogique). Aucun résultat n'est
codé en dur : toutes les métriques sont calculées à l'exécution.
"""

import hmac
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION GÉNÉRALE
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DDoS Shield", page_icon="🛡️", layout="wide")

USERNAME = "testDDoS"
PASSWORD = "havefun"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

# Colonnes retirées avant l'entraînement (identifiants → risque de raccourcis)
COLONNES_EXCLUES = ["timestamp", "src_ip", "dst_ip", "src_port"]
COLONNES_CATEGORIELLES = ["protocol", "dst_port"]

# Modèles qui nécessitent des données standardisées
MODELES_SCALES = ["Régression linéaire", "Régression logistique", "SVM"]

# ─────────────────────────────────────────────────────────────
# 2. STYLE (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg,#0b1220 0%,#0f1b30 100%); }
[data-testid="stSidebar"] { background:#0a1020; border-right:1px solid #1f2d4a; }
h1,h2,h3,h4,p,label,span,div { color:#e6edf7; }
.hero { padding:26px 30px; border-radius:18px; margin-bottom:22px;
        background:linear-gradient(135deg,#0ea5e9 0%,#6366f1 55%,#a855f7 100%); }
.hero h1 { margin:0; font-size:2rem; color:#fff; }
.hero p  { margin:6px 0 0 0; color:#eaf2ff; font-size:1rem; }
.card { background:#111c33; border:1px solid #22345a; border-radius:14px;
        padding:18px 20px; margin-bottom:14px; }
.kpi { background:#111c33; border:1px solid #22345a; border-radius:14px;
       padding:16px; text-align:center; }
.kpi .v { font-size:1.9rem; font-weight:700; color:#38bdf8; }
.kpi .l { font-size:.85rem; color:#94a3b8; }
.alert-red   { background:#3b1219; border:1px solid #ef4444; border-radius:14px;
               padding:20px; text-align:center; }
.alert-green { background:#0f2e22; border:1px solid #22c55e; border-radius:14px;
               padding:20px; text-align:center; }
.alert-red h2, .alert-green h2 { margin:0; }
.stButton>button { border-radius:10px; font-weight:600; }
.warn { background:#2d2410; border:1px solid #f59e0b; border-radius:12px;
        padding:12px 16px; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def hero(icone, titre, sous_titre):
    """Bandeau d'en-tête de page."""
    st.markdown(
        f'<div class="hero"><h1>{icone} {titre}</h1><p>{sous_titre}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(valeur, libelle):
    """Petite carte indicateur."""
    st.markdown(
        f'<div class="kpi"><div class="v">{valeur}</div><div class="l">{libelle}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# 3. AUTHENTIFICATION
# ─────────────────────────────────────────────────────────────
def page_login():
    """Écran de connexion."""
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="hero" style="text-align:center"><h1>🛡️ DDoS Shield</h1>'
            "<p>Plateforme de détection d'attaques par Machine Learning</p></div>",
            unsafe_allow_html=True,
        )
        with st.form("login"):
            st.markdown("### 🔐 Connexion")
            user = st.text_input("👤 Identifiant")
            pwd = st.text_input("🔑 Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter", width="stretch")
        if ok:
            # compare_digest : comparaison à temps constant
            if hmac.compare_digest(user, USERNAME) and hmac.compare_digest(pwd, PASSWORD):
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect.")


if not st.session_state.get("auth", False):
    page_login()
    st.stop()


# ─────────────────────────────────────────────────────────────
# 4. FONCTIONS DU PIPELINE ML (identiques à l'atelier)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def charger_csv(chemin_ou_octets):
    """Charge le CSV (chemin ou contenu téléversé)."""
    if isinstance(chemin_ou_octets, bytes):
        return pd.read_csv(io.BytesIO(chemin_ou_octets))
    return pd.read_csv(chemin_ou_octets)


def pretraiter(df):
    """Étapes 05-07 de l'atelier : doublons puis NaN → 0."""
    n_init = len(df)
    n_nan = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df = df.fillna(0)
    return df, {"lignes_init": n_init, "doublons": n_dup, "nan": n_nan, "lignes_fin": len(df)}


def construire_X(df):
    """Étape 08 : retire les identifiants et encode protocole/port (one-hot)."""
    X = df.drop(columns=[c for c in COLONNES_EXCLUES + ["label"] if c in df.columns])
    return pd.get_dummies(X, columns=COLONNES_CATEGORIELLES, dtype=int)


def entrainer(df, noms_modeles, graine=42):
    """Découpage 70/15/15, standardisation, entraînement, métriques validation."""
    y = df["label"]
    X = construire_X(df)

    # Étape 09 : 70 % train / 15 % validation / 15 % test (stratifié)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=graine, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=graine, stratify=y_tmp)

    # Étape 10 : scaler ajusté UNIQUEMENT sur train (évite la fuite de données)
    scaler = StandardScaler().fit(X_tr)
    S = {"tr": scaler.transform(X_tr), "val": scaler.transform(X_val), "te": scaler.transform(X_te)}
    B = {"tr": X_tr, "val": X_val, "te": X_te}

    catalogue = {
        "Régression linéaire": LinearRegression(),
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=graine),
        "SVM": SVC(kernel="rbf", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=graine, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=graine, n_jobs=2),
    }

    modeles, resultats = {}, []
    for nom in noms_modeles:
        m = catalogue[nom]
        jeu = S if nom in MODELES_SCALES else B
        m.fit(jeu["tr"], y_tr)
        pred = m.predict(jeu["val"])
        if nom == "Régression linéaire":          # sortie continue → seuil 0,5
            pred = (pred >= 0.5).astype(int)
        modeles[nom] = m
        resultats.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, pred),
            "Précision": precision_score(y_val, pred, zero_division=0),
            "Recall": recall_score(y_val, pred, zero_division=0),
            "F1-score": f1_score(y_val, pred, zero_division=0),
        })

    tableau = pd.DataFrame(resultats).sort_values("F1-score", ascending=False).reset_index(drop=True)
    return {
        "modeles": modeles, "tableau": tableau, "scaler": scaler,
        "colonnes": list(X.columns), "S": S, "B": B,
        "y_val": y_val, "y_te": y_te,
        "tailles": (len(X_tr), len(X_val), len(X_te)),
    }


def predire_classes(nom, modele, scaler, colonnes, X_brut):
    """Applique le même prétraitement que l'entraînement, puis prédit."""
    X = X_brut.reindex(columns=colonnes, fill_value=0)   # aligne les colonnes one-hot
    entree = scaler.transform(X) if nom in MODELES_SCALES else X
    sortie = modele.predict(entree)
    if nom == "Régression linéaire":
        score = np.clip(sortie, 0, 1)
        return (sortie >= 0.5).astype(int), score
    proba = modele.predict_proba(entree)[:, 1] if hasattr(modele, "predict_proba") else None
    return sortie.astype(int), proba


def preparer_flux_bruts(df_flux):
    """Prétraite des flux bruts (mêmes étapes que le train) → X encodé."""
    d = df_flux.fillna(0)
    d = d.drop(columns=[c for c in COLONNES_EXCLUES + ["label"] if c in d.columns])
    return pd.get_dummies(d, columns=[c for c in COLONNES_CATEGORIELLES if c in d.columns], dtype=int)


# ─────────────────────────────────────────────────────────────
# 5. BARRE LATÉRALE (navigation + données)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DDoS Shield")
    st.caption("Détection d'attaques par ML")
    page = st.radio(
        "Navigation",
        ["📊 Exploration", "🧹 Prétraitement", "🧠 Entraînement",
         "📈 Métriques & Test", "🎯 Prédiction", "ℹ️ À propos"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**📂 Source des données**")
    fichier = st.file_uploader("Remplacer le dataset (CSV)", type="csv",
                               help="Doit contenir les mêmes colonnes que dataset.csv")
    st.divider()
    st.markdown("👤 Connecté : **testDDoS**")
    if st.button("🚪 Déconnexion", width="stretch"):
        st.session_state.clear()
        st.rerun()

# Chargement du dataset
try:
    df_brut = charger_csv(fichier.getvalue() if fichier else DATA_PATH)
except Exception as e:
    st.error(f"Impossible de charger le dataset : {e}")
    st.stop()

colonnes_requises = {"protocol", "dst_port", "label"}
if not colonnes_requises.issubset(df_brut.columns):
    st.error("Dataset invalide : colonnes `protocol`, `dst_port` et `label` obligatoires.")
    st.stop()

# Prétraitement (rapide, recalculé à chaque rendu)
df, rapport = pretraiter(df_brut)

# ─────────────────────────────────────────────────────────────
# 6. PAGES
# ─────────────────────────────────────────────────────────────
# ---------- 6.1 Exploration ----------
if page.startswith("📊"):
    hero("📊", "Exploration du dataset", "Aperçu des flux réseau étiquetés normal / DDoS")
    st.markdown(
        '<div class="warn">⚠️ Dataset <b>synthétique</b> (pédagogique) : les scores obtenus '
        "ne prouvent pas une efficacité en production.</div><br>",
        unsafe_allow_html=True,
    )
    n1 = int((df_brut["label"] == 0).sum())
    n2 = int((df_brut["label"] == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"{len(df_brut):,}".replace(",", " "), "📦 Flux")
    with c2: kpi(df_brut.shape[1], "🧩 Colonnes")
    with c3: kpi(f"{n1:,}".replace(",", " "), "✅ Trafic normal (0)")
    with c4: kpi(f"{n2:,}".replace(",", " "), "🚨 Attaques DDoS (1)")

    st.markdown("### 🔎 Aperçu des données")
    st.dataframe(df_brut.head(50), width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### ⚖️ Répartition des classes")
        st.bar_chart(df_brut["label"].map({0: "Normal", 1: "DDoS"}).value_counts())
    with g2:
        st.markdown("### 🌐 Protocoles par classe")
        st.bar_chart(pd.crosstab(df_brut["protocol"], df_brut["label"].map({0: "Normal", 1: "DDoS"})))

    st.markdown("### 📐 Moyenne des indicateurs par classe")
    num = df_brut.select_dtypes("number").drop(columns=["src_port", "dst_port"], errors="ignore")
    moy = num.groupby(df_brut["label"].map({0: "Normal", 1: "DDoS"})).mean().drop(columns="label", errors="ignore")
    st.dataframe(moy.T.round(2), width="stretch")
    with st.expander("📋 Statistiques descriptives complètes"):
        st.dataframe(df_brut.describe().T.round(2), width="stretch")

# ---------- 6.2 Prétraitement ----------
elif page.startswith("🧹"):
    hero("🧹", "Prétraitement", "Doublons, valeurs manquantes, sélection des variables, encodage")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(rapport["lignes_init"], "Lignes initiales")
    with c2: kpi(rapport["doublons"], "♻️ Doublons supprimés")
    with c3: kpi(rapport["nan"], "❓ NaN remplacés par 0")
    with c4: kpi(rapport["lignes_fin"], "Lignes conservées")

    st.markdown("### ❓ Valeurs manquantes par colonne (avant traitement)")
    nan_col = df_brut.isna().sum()
    st.dataframe(nan_col[nan_col > 0].rename("NaN").to_frame().T, width="stretch")
    st.markdown(
        '<div class="warn">ℹ️ Comme dans l\'atelier, les NaN sont remplacés par <b>0</b>. '
        "En situation réelle, cela peut créer un signal artificiel : à justifier.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🧬 Variables utilisées par les modèles")
    X_apercu = construire_X(df)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**🚫 Exclues** : `{', '.join(COLONNES_EXCLUES)}`  \n"
                    "Identifiants : risque d'apprendre des raccourcis plutôt que des comportements.")
    with b:
        st.markdown(f"**🔢 Encodées (one-hot)** : `{', '.join(COLONNES_CATEGORIELLES)}`")
    st.write(f"Dimensions de X : **{X_apercu.shape[0]} × {X_apercu.shape[1]}**")
    st.dataframe(X_apercu.head(10), width="stretch")

# ---------- 6.3 Entraînement ----------
elif page.startswith("🧠"):
    hero("🧠", "Entraînement des modèles", "Découpage 70 % / 15 % / 15 % — standardisation sur train uniquement")
    choix = st.multiselect(
        "🤖 Algorithmes à entraîner",
        ["Régression linéaire", "Régression logistique", "SVM", "Random Forest", "XGBoost"],
        default=["Régression logistique", "SVM", "Random Forest", "XGBoost"],
        help="La régression linéaire est une baseline pédagogique (non adaptée à la classification).",
    )
    graine = st.number_input("🎲 random_state", value=42, step=1)
    st.caption("Le SVM peut prendre quelques secondes à plusieurs dizaines de secondes.")

    if st.button("🚀 Lancer l'entraînement", type="primary"):
        if not choix:
            st.warning("Sélectionnez au moins un algorithme.")
        else:
            with st.spinner("Entraînement en cours..."):
                st.session_state["res"] = entrainer(df, choix, int(graine))
            st.success("✅ Entraînement terminé.")

    if "res" in st.session_state:
        r = st.session_state["res"]
        t1, t2, t3 = r["tailles"]
        c1, c2, c3 = st.columns(3)
        with c1: kpi(t1, "🏋️ Train (70 %)")
        with c2: kpi(t2, "🧪 Validation (15 %)")
        with c3: kpi(t3, "🔒 Test (15 %)")
        st.markdown("### 🏆 Résultats sur validation")
        st.dataframe(r["tableau"].round(4), width="stretch")
        st.info("Rendez-vous dans « 📈 Métriques & Test » pour les graphiques et l'évaluation finale.")
    else:
        st.info("Aucun modèle entraîné pour le moment.")

# ---------- 6.4 Métriques & Test ----------
elif page.startswith("📈"):
    hero("📈", "Métriques & évaluation finale", "Accuracy, précision, recall, F1 et matrice de confusion")
    if "res" not in st.session_state:
        st.warning("⚠️ Entraînez d'abord les modèles (page « 🧠 Entraînement »).")
        st.stop()
    r = st.session_state["res"]
    tab = r["tableau"]

    st.markdown("### 📊 Comparaison sur validation")
    st.dataframe(tab.round(4), width="stretch")
    st.bar_chart(tab.set_index("Modèle")[["Accuracy", "Précision", "Recall", "F1-score"]])

    st.markdown("### 🧪 Évaluation sur le jeu de test (modèle choisi sur validation)")
    meilleur = tab.iloc[0]["Modèle"]
    nom = st.selectbox("Modèle à évaluer", list(r["modeles"].keys()),
                       index=list(r["modeles"].keys()).index(meilleur))
    st.caption(f"Meilleur F1 sur validation : **{meilleur}**. "
               "Le test doit servir une seule fois, pour le modèle retenu final.")

    jeu = r["S"] if nom in MODELES_SCALES else r["B"]
    pred = r["modeles"][nom].predict(jeu["te"])
    if nom == "Régression linéaire":
        pred = (pred >= 0.5).astype(int)
    y_te = r["y_te"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"{accuracy_score(y_te, pred):.4f}", "🎯 Accuracy")
    with c2: kpi(f"{precision_score(y_te, pred, zero_division=0):.4f}", "🔔 Précision")
    with c3: kpi(f"{recall_score(y_te, pred, zero_division=0):.4f}", "📡 Recall")
    with c4: kpi(f"{f1_score(y_te, pred, zero_division=0):.4f}", "⚖️ F1-score")

    st.markdown("### 🧮 Matrice de confusion (test)")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Normal", "DDoS"],
                                            cmap="Blues", ax=ax)
    ax.set_title(nom)
    left, _ = st.columns([1, 1])
    with left:
        st.pyplot(fig)
    st.caption("Faux négatifs = attaques non détectées · Faux positifs = trafic normal signalé à tort.")

    if nom in ("Random Forest", "XGBoost"):
        st.markdown("### 🔬 Variables les plus importantes")
        imp = pd.Series(r["modeles"][nom].feature_importances_, index=r["colonnes"])
        st.bar_chart(imp.sort_values(ascending=False).head(10))

# ---------- 6.5 Prédiction ----------
elif page.startswith("🎯"):
    hero("🎯", "Prédiction d'attaque", "Analyse d'un flux unique ou d'un fichier CSV complet")
    if "res" not in st.session_state:
        st.warning("⚠️ Entraînez d'abord les modèles (page « 🧠 Entraînement »).")
        st.stop()
    r = st.session_state["res"]
    nom = st.selectbox("🤖 Modèle utilisé", list(r["modeles"].keys()),
                       index=list(r["modeles"].keys()).index(r["tableau"].iloc[0]["Modèle"]))
    modele = r["modeles"][nom]

    onglet1, onglet2 = st.tabs(["🔍 Flux unique", "📁 Fichier CSV"])

    # --- Flux unique ---
    with onglet1:
        st.markdown("Saisissez les caractéristiques du flux. Les débits et la taille moyenne "
                    "sont **calculés automatiquement** (même formules que le dataset).")
        a, b, c = st.columns(3)
        with a:
            protocole = st.selectbox("🌐 Protocole", sorted(df["protocol"].unique()))
            port = st.selectbox("🔌 Port destination", sorted(df["dst_port"].unique()))
            duree = st.number_input("⏱️ Durée (ms)", min_value=1.0, value=500.0, step=10.0)
        with b:
            paquets = st.number_input("📦 Nombre de paquets", min_value=0, value=100, step=10)
            octets = st.number_input("💾 Nombre d'octets", min_value=0, value=40000, step=1000)
            syn = st.number_input("🔺 Paquets SYN", min_value=0, value=0, step=1)
        with c:
            ack = st.number_input("✔️ Paquets ACK", min_value=0, value=0, step=1)
            conns = st.number_input("🔗 Connexions (10 s)", min_value=0, value=50, step=5)
            sources = st.number_input("🧑‍💻 Sources uniques (10 s)", min_value=0, value=20, step=1)

        if st.button("⚡ Analyser le flux", type="primary"):
            sec = duree / 1000
            flux = pd.DataFrame([{
                "protocol": protocole, "dst_port": port, "duration_ms": duree,
                "packet_count": paquets, "byte_count": octets,
                "packets_per_second": paquets / sec, "bytes_per_second": octets / sec,
                "avg_packet_size_bytes": octets / paquets if paquets else 0,
                "syn_count": syn, "ack_count": ack,
                "connections_10s": conns, "unique_sources_10s": sources,
            }])
            cls, score = predire_classes(nom, modele, r["scaler"], r["colonnes"], preparer_flux_bruts(flux))
            if cls[0] == 1:
                st.markdown('<div class="alert-red"><h2>🚨 Attaque DDoS détectée</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-green"><h2>✅ Trafic normal</h2></div>', unsafe_allow_html=True)
            if score is not None:
                lib = "Score du modèle linéaire (borné 0-1)" if nom == "Régression linéaire" \
                    else "Probabilité d'attaque estimée"
                st.progress(float(score[0]), text=f"{lib} : {score[0]*100:.1f} %")
            st.caption(f"Débit calculé : {paquets/sec:,.1f} paquets/s · {octets/sec:,.0f} octets/s")

    # --- Fichier CSV ---
    with onglet2:
        st.markdown("Chargez un CSV de flux (mêmes colonnes que le dataset ; `label` facultatif).")
        up = st.file_uploader("📁 Fichier de flux", type="csv", key="batch")
        if up is not None:
            try:
                lot = pd.read_csv(up)
                attendues = set(df.columns) - {"label"} - set(COLONNES_EXCLUES)
                manquantes = attendues - set(lot.columns)
                if manquantes:
                    st.error(f"Colonnes manquantes : {', '.join(sorted(manquantes))}")
                else:
                    cls, score = predire_classes(nom, modele, r["scaler"], r["colonnes"], preparer_flux_bruts(lot))
                    sortie = lot.copy()
                    sortie["prediction"] = np.where(cls == 1, "DDoS", "Normal")
                    if score is not None:
                        sortie["score_attaque"] = np.round(score, 4)
                    c1, c2 = st.columns(2)
                    with c1: kpi(int((cls == 1).sum()), "🚨 Flux détectés comme DDoS")
                    with c2: kpi(int((cls == 0).sum()), "✅ Flux détectés comme normaux")
                    st.dataframe(sortie.head(200), width="stretch")
                    if "label" in lot.columns:
                        st.caption(f"Accuracy sur ce fichier (labels fournis) : "
                                   f"{accuracy_score(lot['label'], cls):.4f}")
                    st.download_button("⬇️ Télécharger les résultats (CSV)",
                                       sortie.to_csv(index=False).encode("utf-8"),
                                       "predictions_ddos.csv", "text/csv")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")

# ---------- 6.6 À propos ----------
else:
    hero("ℹ️", "À propos", "Méthodologie et limites")
    st.markdown(
        """
<div class="card">

**🔁 Pipeline** : chargement CSV → suppression des doublons → NaN remplacés par 0 →
exclusion des identifiants (`timestamp`, IP, `src_port`) → encodage one-hot (`protocol`, `dst_port`) →
découpage 70/15/15 stratifié → standardisation (fit sur train) → entraînement → métriques.

**🤖 Modèles** : régression linéaire (baseline), régression logistique, SVM (RBF), Random Forest, XGBoost.

**⚠️ Limites** : dataset synthétique ; les performances ne se transposent pas telles quelles à un réseau réel.
Les identifiants de connexion sont codés en dur dans le code : adaptés à une démonstration uniquement.

</div>
""",
        unsafe_allow_html=True,
    )
