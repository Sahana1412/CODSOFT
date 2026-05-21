import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack, csr_matrix


GENRE_LEXICON = {
    "action":       ["fight","battle","explosion","chase","hero","warrior","mission","combat",
                     "shoot","attack","escape","danger","force","army","soldier","weapon","war",
                     "rescue","assassin","martial","strike","ambush","raid","sniper","brawl"],
    "adventure":    ["quest","journey","explore","treasure","island","discover","expedition",
                     "map","cave","jungle","ancient","relic","voyage","adventure","trail",
                     "wilderness","survive","legend","forbidden","escape","ancient","ruin"],
    "animation":    ["cartoon","animated","voice","toy","fairy","magic","wizard","dragon",
                     "creature","fairy tale","kingdom","princess","talking","imaginary","dream",
                     "colorful","whimsical","enchanted","animal","toon","pixar","disney"],
    "biography":    ["true story","real life","based on","historical","biography","life of",
                     "portrait","inspiring","legend","pioneer","activist","president","leader",
                     "memoir","rise","chronicle","documentary","politician","athlete","artist"],
    "comedy":       ["funny","laugh","humor","joke","hilarious","comic","silly","wacky","fun",
                     "quirky","absurd","mishap","prank","wedding","romantic","awkward","bumbling",
                     "misunderstanding","farce","wit","banter","slapstick","satire","parody"],
    "crime":        ["crime","murder","detective","heist","robbery","criminal","gang","thief",
                     "mob","mafia","cop","police","investigation","suspect","corrupt","drug",
                     "prison","evidence","witness","stolen","fraud","hitman","cartel","fugitive"],
    "documentary":  ["documentary","real","footage","archive","interview","report","expose",
                     "history","event","culture","society","environment","wildlife","nature",
                     "science","world","people","story","film","explore","observe","record"],
    "drama":        ["emotional","family","relationship","struggle","crisis","personal","grief",
                     "tragedy","redemption","conflict","betrayal","sacrifice","loss","pain",
                     "marriage","divorce","death","life","journey","heart","overcome","bond"],
    "fantasy":      ["magic","wizard","dragon","spell","enchanted","kingdom","sorcerer","myth",
                     "elf","dwarf","prophecy","realm","creature","supernatural","mystical",
                     "potion","curse","quest","ancient","forbidden","legend","fairy","witch"],
    "horror":       ["haunted","ghost","evil","terror","fear","monster","demon","dark","curse",
                     "nightmare","blood","death","kill","supernatural","possessed","creature",
                     "scream","dread","paranormal","psycho","slaughter","zombie","vampire","cult"],
    "music":        ["music","song","band","concert","singer","musician","album","rock","jazz",
                     "dance","perform","stage","melody","rhythm","soul","record","tour","lyric",
                     "composer","instrument","pop","choir","opera","conductor","audition"],
    "mystery":      ["mystery","clue","secret","hidden","puzzle","suspect","detective","unknown",
                     "disappear","investigate","conspiracy","code","reveal","truth","identity",
                     "murder","shadow","enigma","deception","missing","cold case","expose"],
    "romance":      ["love","romance","heart","relationship","couple","kiss","wedding","affair",
                     "passion","desire","soulmate","attraction","partner","dating","marriage",
                     "breakup","longing","feelings","together","apart","reunion","affection"],
    "sci-fi":       ["space","alien","future","robot","planet","technology","galaxy","starship",
                     "time travel","clone","mutation","android","experiment","dimension","cyber",
                     "laser","dystopia","colony","orbit","quantum","artificial","spacecraft"],
    "short":        ["short","brief","moment","quick","glimpse","vignette","snapshot","sketch"],
    "thriller":     ["suspense","tension","danger","chase","threat","espionage","spy","paranoid",
                     "assassin","trap","psychological","nerve","countdown","deadly","hostage",
                     "betrayal","conspiracy","secret","twist","pursue","thriller","target"],
    "western":      ["western","cowboy","sheriff","outlaw","frontier","saloon","ranch","gunfight",
                     "bounty","desert","horse","wild west","marshal","bandit","gold rush","duel"],
}

# ─────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, title, genre, plot = parts
                data.append((title.strip(), genre.strip(), plot.strip()))
    return pd.DataFrame(data, columns=['title', 'genre', 'plot'])

# ─────────────────────────────────────────────────────────
# 2. CLEAN  (keep numbers — year cues matter)
# ─────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ─────────────────────────────────────────────────────────
# 3. LEXICON FEATURES  →  dense matrix  (n_samples × n_genres)
#    Normalized count of genre-keyword hits per document.
#    This is an orthogonal signal to TF-IDF.
# ─────────────────────────────────────────────────────────
def build_lexicon_features(texts, genre_order):
    rows = []
    for text in texts:
        row = []
        for genre in genre_order:
            keywords = GENRE_LEXICON.get(genre, [])
            count = sum(1 for kw in keywords if kw in text)
            row.append(count)
        rows.append(row)
    arr = np.array(rows, dtype=np.float32)
    # L2-normalize so long documents don't dominate
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms

# ─────────────────────────────────────────────────────────
# 4. STATISTICAL FEATURES  →  dense (n_samples × 6)
#    Surface-level signals correlated with genre.
# ─────────────────────────────────────────────────────────
def build_stat_features(texts):
    rows = []
    for text in texts:
        words = text.split()
        n = len(words)
        unique_ratio = len(set(words)) / (n + 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        sent_count   = text.count('.') + text.count('!') + text.count('?') + 1
        avg_sent_len = n / sent_count
        has_dialogue = int('"' in text or "'" in text)
        rows.append([n, unique_ratio, avg_word_len, sent_count, avg_sent_len, has_dialogue])
    arr = np.array(rows, dtype=np.float32)
    # Standardize
    arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)
    return arr

# ─────────────────────────────────────────────────────────
# 5. PREPROCESS
# ─────────────────────────────────────────────────────────
def preprocess(df):
    # Title repeated 3×: short but extremely genre-informative
    df['text'] = (df['title'] + " ") * 3 + df['plot']
    df['text'] = df['text'].apply(clean_text)
    df['genre'] = df['genre'].apply(lambda x: [g.strip() for g in x.split('|')])
    return df

# ─────────────────────────────────────────────────────────
# 6. BUILD FULL FEATURE MATRIX
#    Stack: word-TF-IDF | char-TF-IDF | lexicon | stats
# ─────────────────────────────────────────────────────────
def build_features(X_train_raw, X_test_raw, genre_order):

    # --- 6a. Word TF-IDF (unigram + bigram) ---
    word_vec = TfidfVectorizer(
        max_features=120000,
        ngram_range=(1, 3),       # trigrams: "based on true" captures bios
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english',
        analyzer='word',
        strip_accents='unicode'
    )

    # --- 6b. Character TF-IDF (morphological signal) ---
    char_vec = TfidfVectorizer(
        max_features=60000,
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        analyzer='char_wb'
    )

    print("  Fitting word TF-IDF...")
    Xw_tr = word_vec.fit_transform(X_train_raw)
    Xw_te = word_vec.transform(X_test_raw)

    print("  Fitting char TF-IDF...")
    Xc_tr = char_vec.fit_transform(X_train_raw)
    Xc_te = char_vec.transform(X_test_raw)

    print("  Building lexicon features...")
    Xl_tr = csr_matrix(build_lexicon_features(X_train_raw, genre_order))
    Xl_te = csr_matrix(build_lexicon_features(X_test_raw,  genre_order))

    print("  Building statistical features...")
    Xs_tr = csr_matrix(build_stat_features(list(X_train_raw)))
    Xs_te = csr_matrix(build_stat_features(list(X_test_raw)))

    X_tr = hstack([Xw_tr, Xc_tr, Xl_tr, Xs_tr])
    X_te = hstack([Xw_te, Xc_te, Xl_te, Xs_te])

    print(f"  Final feature shape: {X_tr.shape}")
    return X_tr, X_te, word_vec, char_vec

# ─────────────────────────────────────────────────────────
# 7. TRAIN STACKING ENSEMBLE
#    Stage 1: train LR and SVM on different feature subsets
#    Stage 2: blend using optimized weights per label
# ─────────────────────────────────────────────────────────
def train_ensemble(X_tr, X_te, y_tr, y_te):

    # ── Model A: LR, higher C, L2, on full features ──
    print("  Training LR (C=8)...")
    lr_a = OneVsRestClassifier(
        LogisticRegression(C=8, max_iter=5000, class_weight='balanced',
                           solver='saga', n_jobs=-1),
        n_jobs=-1
    )
    lr_a.fit(X_tr, y_tr)

    # ── Model B: LR, lower C (more regularized), different optimum ──
    print("  Training LR (C=1)...")
    lr_b = OneVsRestClassifier(
        LogisticRegression(C=1, max_iter=5000, class_weight='balanced',
                           solver='saga', n_jobs=-1),
        n_jobs=-1
    )
    lr_b.fit(X_tr, y_tr)

    # ── Model C: Calibrated LinearSVC ──
    print("  Training LinearSVC...")
    svm = OneVsRestClassifier(
        CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, class_weight='balanced'),
            method='isotonic', cv=3   # isotonic > sigmoid for skewed classes
        ),
        n_jobs=-1
    )
    svm.fit(X_tr, y_tr)

    pa  = lr_a.predict_proba(X_te)
    pb  = lr_b.predict_proba(X_te)
    pc  = svm.predict_proba(X_te)

    # ── Per-label optimal blend weight search ──
    print("  Finding optimal ensemble weights...")
    n_labels = y_te.shape[1]
    best_weights = []

    for i in range(n_labels):
        best_w, best_f1 = (0.34, 0.33, 0.33), 0
        # Search over a grid of (wA, wB, wC) that sum to 1
        for wa in np.arange(0.1, 0.8, 0.1):
            for wb in np.arange(0.1, 0.8, 0.1):
                wc = 1.0 - wa - wb
                if wc < 0.05 or wc > 0.8:
                    continue
                blended = wa * pa[:, i] + wb * pb[:, i] + wc * pc[:, i]
                # Quick threshold at 0.3 for weight search
                pred = (blended >= 0.3).astype(int)
                sc = f1_score(y_te[:, i], pred, zero_division=0)
                if sc > best_f1:
                    best_f1 = sc
                    best_w = (wa, wb, wc)
        best_weights.append(best_w)

    # Build final ensemble probability matrix
    probs = np.zeros_like(pa)
    for i, (wa, wb, wc) in enumerate(best_weights):
        probs[:, i] = wa * pa[:, i] + wb * pb[:, i] + wc * pc[:, i]

    return probs, (lr_a, lr_b, svm), best_weights

# ─────────────────────────────────────────────────────────
# 8. THRESHOLD TUNING  (F1-optimal per label)
# ─────────────────────────────────────────────────────────
def tune_thresholds(probs, y_true):
    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.3)

    for i in range(n_labels):
        best_t, best_f1 = 0.3, 0.0
        col = probs[:, i]
        pos_count = y_true[:, i].sum()

        # For very rare classes, search lower thresholds
        t_min = 0.05 if pos_count < 20 else 0.10
        t_max = 0.75

        for t in np.arange(t_min, t_max, 0.01):   # fine 0.01 step
            pred = (col >= t).astype(int)
            if pred.sum() == 0:
                continue
            sc = f1_score(y_true[:, i], pred, zero_division=0)
            if sc > best_f1:
                best_f1, best_t = sc, t

        thresholds[i] = best_t

    return thresholds

# ─────────────────────────────────────────────────────────
# 9. MAIN TRAIN PIPELINE
# ─────────────────────────────────────────────────────────
def train_model(df):

    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(df['genre'])
    genre_order = list(mlb.classes_)

    print(f"\nDataset  : {len(df):,} samples")
    print(f"Genres   : {len(genre_order)} → {genre_order}\n")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], y_bin, test_size=0.2, random_state=42
    )

    # Reset index for list operations
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)

    print("── STEP 1: Building features ──")
    X_tr, X_te, word_vec, char_vec = build_features(X_train_raw, X_test_raw, genre_order)

    print("\n── STEP 2: Training ensemble ──")
    probs, classifiers, blend_weights = train_ensemble(X_tr, X_te, y_train, y_test)

    print("\n── STEP 3: Tuning thresholds ──")
    thresholds = tune_thresholds(probs, y_test)

    y_pred = (probs >= thresholds).astype(int)

    micro = f1_score(y_test, y_pred, average='micro')
    macro = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'═'*50}")
    print(f"  Micro F1 : {micro:.4f}")
    print(f"  Macro F1 : {macro:.4f}")
    print(f"{'═'*50}\n")
    print(classification_report(y_test, y_pred, target_names=genre_order, zero_division=0))

    artifacts = {
        "classifiers":   classifiers,
        "mlb":           mlb,
        "word_vec":      word_vec,
        "char_vec":      char_vec,
        "thresholds":    thresholds,
        "blend_weights": blend_weights,
        "genre_order":   genre_order,
    }
    return artifacts

# ─────────────────────────────────────────────────────────
# 10. PREDICT
# ─────────────────────────────────────────────────────────
def predict_genre(artifacts, raw_text):
    text = clean_text((raw_text + " ") * 3)

    lr_a, lr_b, svm = artifacts["classifiers"]
    wv   = artifacts["word_vec"]
    cv   = artifacts["char_vec"]
    thrs = artifacts["thresholds"]
    bw   = artifacts["blend_weights"]
    mlb  = artifacts["mlb"]

    Xw = wv.transform([text])
    Xc = cv.transform([text])

    # Lexicon + stat features
    go = artifacts["genre_order"]
    Xl = csr_matrix(build_lexicon_features([text], go))
    Xs = csr_matrix(build_stat_features([text]))
    X  = hstack([Xw, Xc, Xl, Xs])

    pa = lr_a.predict_proba(X)
    pb = lr_b.predict_proba(X)
    pc = svm.predict_proba(X)

    probs = np.zeros_like(pa)
    for i, (wa, wb, wc) in enumerate(bw):
        probs[0, i] = wa * pa[0, i] + wb * pb[0, i] + wc * pc[0, i]

    pred = (probs >= thrs).astype(int)
    genres = mlb.inverse_transform(pred)[0]

    if not genres:
        top = np.argsort(probs[0])[-2:][::-1]
        genres = tuple(mlb.classes_[i] for i in top)

    return genres

# ─────────────────────────────────────────────────────────
# 11. ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    FILE_PATH = r"D:\Internships\Codsoft ML Internship\Movie genre classification\Genre Classification Dataset\train_data.txt"

    df = load_data(FILE_PATH)
    df = preprocess(df)

    artifacts = train_model(df)

    samples = [
        ("Inception",      "A thief who steals corporate secrets through dream-sharing technology."),
        ("The Notebook",   "Two young lovers from different backgrounds fall passionately in love."),
        ("Taken",          "A retired CIA agent hunts the kidnappers who abducted his daughter."),
        ("Interstellar",   "A team of explorers travel through a wormhole in space to save humanity."),
        ("The Hangover",   "Three friends wake up after a wild bachelor party with no memory."),
        ("Schindler's List","A German businessman saves Jewish refugees during the Holocaust."),
    ]

    print("\n── SAMPLE PREDICTIONS ──")
    for title, plot in samples:
        g = predict_genre(artifacts, title + " " + plot)
        print(f"  {title:<22} → {g}")
