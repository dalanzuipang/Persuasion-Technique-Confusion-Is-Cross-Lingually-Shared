"""
Shared utilities for:
  "Aggregation Granularity Reverses Confusion Diagnostics:
   Persuasion Technique Confusion Is Cross-Lingually Shared"

Every notebook imports from here, so the loading rules, the TCM, the
bi-normalisation and all hyper-parameters exist in exactly one place.

Usage inside a notebook:
    import sys; sys.path.insert(0, '../src')
    from common import *
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths (relative to repository root)
# ------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / 'data'
ARTICLES  = DATA / 'articles'      # data/articles/{en,pl,ru}/article*.txt
GOLD_DIR  = DATA / 'gold'          # data/gold/{en,pl,ru}/labels-subtask-3-spans.txt
PRED_DIR  = ROOT / 'predictions'   # predictions/{lang}_{method}.tsv
MATRICES  = ROOT / 'matrices'
RESULTS   = ROOT / 'results'
FIGURES   = RESULTS / 'figures'

for _p in (MATRICES / 'article', MATRICES / 'paragraph',
           MATRICES / 'paragraph_intersect', RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Configuration (paper §III)
# ------------------------------------------------------------------
LANGS   = ['en', 'pl', 'ru']
METHODS = ['sup_ft', 'prompt_a', 'iter_ens']
CONFIGS = [(l, m) for l in LANGS for m in METHODS]

LANG_LABEL   = {'en': 'EN', 'pl': 'PL', 'ru': 'RU'}
METHOD_LABEL = {'sup_ft': 'Sup-FT', 'prompt_a': 'Prompt-A', 'iter_ens': 'Iter-Ens'}

# §III-C: paragraph segmentation must match the detectors' preprocessing
KEEP_LONG    = False    # False => drop paragraphs longer than MAX_PARA_LEN
MAX_PARA_LEN = 1000

# §III-B: support floor
MIN_SUP = 1.0

# §III-B: IPF
IPF_TOL     = 1e-7
IPF_EPS     = 1e-8
IPF_MAXITER = 100_000
IPF_DEMO_CAP = 2000     # cap used for the pathology demo in notebook 07

# §IV-B: robust-pair threshold
THRESH = 0.15

# §IV-B: resampling
B_BOOT = 5000
SEED   = 42

# ------------------------------------------------------------------
# Expected values (asserted by the notebooks; see paper §III-D)
# ------------------------------------------------------------------
EXPECT_ARTICLES   = {'en': 90,   'pl': 49,  'ru': 48}
EXPECT_PARAGRAPHS = {'en': 3127, 'pl': 779, 'ru': 503}
EXPECT_GOLD_SPANS = {'en': 1801, 'pl': 985, 'ru': 739}   # PL after de-duplication

# ------------------------------------------------------------------
# Taxonomy
# ------------------------------------------------------------------
TECHNIQUES = [l.strip() for l in
              (DATA / 'techniques_subtask3.txt').read_text(encoding='utf-8').splitlines()
              if l.strip()]
TECH_SET  = set(TECHNIQUES)
TECH2IDX  = {t: i for i, t in enumerate(TECHNIQUES)}
C         = len(TECHNIQUES)
assert C == 23, f'expected 23 techniques, found {C}'

# Aristotelian super-classes (paper §IV-D). Used only for post-hoc
# interpretation; does not enter the transport cost.
SUPERCLASS = {
    'Ethos': ['Appeal_to_Authority', 'Appeal_to_Hypocrisy', 'Doubt',
              'Guilt_by_Association', 'Name_Calling-Labeling',
              'Questioning_the_Reputation'],
    'Logos': ['Causal_Oversimplification', 'Consequential_Oversimplification',
              'False_Dilemma-No_Choice', 'False_Equivalence',
              'Obfuscation-Vagueness-Confusion', 'Red_Herring', 'Straw_Man',
              'Whataboutism'],
    'Pathos': ['Appeal_to_Fear-Prejudice', 'Appeal_to_Pity',
               'Appeal_to_Popularity', 'Appeal_to_Time', 'Appeal_to_Values',
               'Conversation_Killer', 'Exaggeration-Minimisation',
               'Flag_Waving', 'Loaded_Language', 'Repetition', 'Slogans'],
}
TECH2SUPER = {t: s for s, ts in SUPERCLASS.items() for t in ts}

ABBR = {
    'Appeal_to_Authority': 'AtAu', 'Appeal_to_Popularity': 'AtPo',
    'Appeal_to_Values': 'AtV', 'Appeal_to_Fear-Prejudice': 'AFP',
    'Flag_Waving': 'FW', 'Causal_Oversimplification': 'CO',
    'False_Dilemma-No_Choice': 'FDNC', 'Consequential_Oversimplification': 'ConO',
    'Straw_Man': 'SM', 'Red_Herring': 'RH', 'Whataboutism': 'Wh',
    'Slogans': 'Sl', 'Appeal_to_Time': 'AtT', 'Conversation_Killer': 'CK',
    'Loaded_Language': 'LL', 'Repetition': 'Rep',
    'Exaggeration-Minimisation': 'EM',
    'Obfuscation-Vagueness-Confusion': 'OVC',
    'Name_Calling-Labeling': 'NCL', 'Doubt': 'Db',
    'Guilt_by_Association': 'GbA', 'Appeal_to_Hypocrisy': 'AtH',
    'Questioning_the_Reputation': 'QtR', 'Appeal_to_Pity': 'AtPi',
    'False_Equivalence': 'FE',
}

# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def normalize_id(aid):
    """Sup-FT predictions carry an 'article' prefix; gold does not."""
    return str(aid).replace('article', '').strip()


def build_para_index(lang, keep_long=KEEP_LONG, max_len=MAX_PARA_LEN):
    """
    {article_id: [(line_idx, char_start, char_end, text), ...]}

    Segmentation rule (paper §III-C), matching the detectors' own
    preprocessing: content.splitlines(), drop blank lines, drop lines
    longer than `max_len`. Offsets are character-level, half-open.
    """
    idx = {}
    for f in sorted((ARTICLES / lang).iterdir()):
        if not f.is_file():
            continue
        aid = normalize_id(f.stem)
        if not aid.isdigit():
            continue
        paras, off = [], 0
        for li, line in enumerate(f.read_text(encoding='utf-8').splitlines()):
            s, e = off, off + len(line)
            off = e + 1                      # +1 for the newline
            if not line.strip():
                continue
            if (not keep_long) and len(line) > max_len:
                continue
            paras.append((li, s, e, line))
        idx[aid] = paras
    return idx


def load_spans(path, lang, dedup=True):
    """
    Read a 4-column span TSV: article_id, technique, start, end.

    PL note: the CheckThat! 2024 Polish dev span file ships with every
    row duplicated (1,970 rows = 985 unique x 2). We de-duplicate on
    (article_id, technique, start, end). All PL numbers in the paper use
    the de-duplicated gold.
    """
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['article_id', 'technique', 'start', 'end'],
                     dtype=str, keep_default_na=False)
    df = df[df['technique'].isin(TECH_SET)]
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end']   = pd.to_numeric(df['end'],   errors='coerce')
    df = df.dropna(subset=['start', 'end']).copy()
    df['start'] = df['start'].astype(int)
    df['end']   = df['end'].astype(int)
    df['article_id'] = df['article_id'].apply(normalize_id)
    if dedup:
        df = df.drop_duplicates(subset=['article_id', 'technique', 'start', 'end'])
    return df.reset_index(drop=True)


def gold_path(lang):
    return GOLD_DIR / lang / 'labels-subtask-3-spans.txt'


def pred_path(lang, method):
    return PRED_DIR / f'{lang}_{method}.tsv'


def load_gold(lang):
    return load_spans(gold_path(lang), lang, dedup=True)


def load_pred(lang, method):
    return load_spans(pred_path(lang, method), lang, dedup=True)


# ------------------------------------------------------------------
# Multi-hot construction at two granularities (paper §III-C)
# ------------------------------------------------------------------
def article_multihot(gold_df, pred_df, article_ids):
    """Rows = articles. A technique is present if annotated / predicted
    anywhere in the article. Articles with no gold contribute an all-zero
    row and are excluded by the TCM rule of §III-A."""
    ids = sorted(article_ids)
    Yg = np.zeros((len(ids), C))
    Yp = np.zeros((len(ids), C))
    g = gold_df.groupby('article_id')['technique'].apply(set).to_dict()
    p = pred_df.groupby('article_id')['technique'].apply(set).to_dict()
    for r, aid in enumerate(ids):
        for t in g.get(aid, ()):
            Yg[r, TECH2IDX[t]] = 1.
        for t in p.get(aid, ()):
            Yp[r, TECH2IDX[t]] = 1.
    return Yg, Yp, ids


def spans_to_paragraphs(df, para_idx):
    """{(article_id, line_idx): set(techniques)}. A span is assigned to
    every paragraph it overlaps (half-open intervals)."""
    out = defaultdict(set)
    n_unmatched = 0
    for aid, sub in df.groupby('article_id'):
        paras = para_idx.get(aid)
        if paras is None:
            n_unmatched += len(sub)
            continue
        for _, r in sub.iterrows():
            hit = False
            for (li, ps, pe, _txt) in paras:
                if r['start'] < pe and ps < r['end']:
                    out[(aid, li)].add(r['technique'])
                    hit = True
            if not hit:
                n_unmatched += 1
    return dict(out), n_unmatched


def paragraph_keys(para_idx):
    return [(aid, li) for aid, ps in para_idx.items() for (li, _s, _e, _t) in ps]


def paragraph_multihot(gold_ps, pred_ps, keys):
    Yg = np.zeros((len(keys), C))
    Yp = np.zeros((len(keys), C))
    for i, k in enumerate(keys):
        for t in gold_ps.get(k, ()):
            Yg[i, TECH2IDX[t]] = 1.
        for t in pred_ps.get(k, ()):
            Yp[i, TECH2IDX[t]] = 1.
    return Yg, Yp


# ------------------------------------------------------------------
# TCM (paper §III-A, Eq. 1-2) -- Erbani et al., uniform cost
# ------------------------------------------------------------------
def tcm_contribution(y, yh):
    """Per-unit transport plan under uniform cost.
    Diagonal keeps min(u_k, v_k); the residual deficit d couples to the
    residual excess e as the rank-one outer product d e^T / ||d||_1."""
    ny, nh = y.sum(), yh.sum()
    if ny == 0 or nh == 0:
        return np.zeros((len(y), len(y)))
    u, v = y / ny, yh / nh
    m       = np.minimum(u, v)
    deficit = u - m
    excess  = v - m
    total   = deficit.sum()
    pi_diag = np.diag(m)
    if total < 1e-10:
        return pi_diag                       # perfect prediction
    pi_off = np.outer(deficit, excess) / total
    np.fill_diagonal(pi_off, 0.0)
    return pi_diag + pi_off


def compute_tcm(Yg, Yp):
    """TCM = sum_n pi_n. Units with zero gold or zero predicted mass are
    excluded (uniform-cost OT is undefined there). Returns (T, n_used)."""
    T, n = np.zeros((C, C)), 0
    for i in range(len(Yg)):
        if Yg[i].sum() == 0 or Yp[i].sum() == 0:
            continue
        T += tcm_contribution(Yg[i], Yp[i])
        n += 1
    return T, n


def residual_support(Yg, Yp):
    """Per-unit support of Eq. (2), averaged over the units that enter the
    TCM. Reproduces Table III."""
    d0, e0, cells, gold, pred = [], [], [], [], []
    for i in range(len(Yg)):
        y, yh = Yg[i], Yp[i]
        ny, nh = y.sum(), yh.sum()
        if ny == 0 or nh == 0:
            continue
        u, v = y / ny, yh / nh
        m = np.minimum(u, v)
        nd = int(((u - m) > 1e-12).sum())
        ne = int(((v - m) > 1e-12).sum())
        d0.append(nd); e0.append(ne); cells.append(nd * ne)
        gold.append(ny); pred.append(nh)
    return dict(n_units=len(d0),
                gold=float(np.mean(gold)), pred=float(np.mean(pred)),
                over_pred=float(np.mean(pred) / np.mean(gold)),
                d0=float(np.mean(d0)), e0=float(np.mean(e0)),
                cells=float(np.mean(cells)))


# ------------------------------------------------------------------
# IPF bi-normalisation (paper §III-B, Eq. 3)
# ------------------------------------------------------------------
def bi_norm(M, eps=IPF_EPS, maxiter=IPF_MAXITER, tol=IPF_TOL):
    """Alternating row/column normalisation to a doubly-stochastic matrix.
    Returns (Q, iterations, converged). `converged` is False if the cap was
    hit -- notebook 07 relies on this."""
    Q = M.copy().astype(float) + eps
    for it in range(maxiter):
        Q /= Q.sum(1, keepdims=True)
        Q /= Q.sum(0, keepdims=True)
        err = max(abs(Q.sum(1) - 1).max(), abs(Q.sum(0) - 1).max())
        if err < tol:
            return Q, it + 1, True
    return Q, maxiter, False


def per_class_marginals(T):
    """Row sum = gold mass, column sum = predicted mass, per technique."""
    return pd.DataFrame({'technique': TECHNIQUES,
                         'gold_count': T.sum(1),
                         'pred_count': T.sum(0)})


def support_floor_keep(T, s=MIN_SUP):
    """Techniques with both gold and predicted mass >= s in this matrix."""
    m = per_class_marginals(T)
    return set(m[(m['gold_count'] >= s) & (m['pred_count'] >= s)]['technique'])


def common_subspace(raw_dict, s=MIN_SUP):
    """Techniques clearing the floor in ALL nine configurations, in
    taxonomy order. Restricting to a common index set is a precondition
    for the rank correlations of §IV-B."""
    keeps = [support_floor_keep(T, s) for T in raw_dict.values()]
    common = sorted(set.intersection(*keeps), key=lambda t: TECH2IDX[t])
    return common, [TECH2IDX[t] for t in common]


def offdiag_vector(Tb, K):
    """Symmetrised off-diagonal scores, max(A->B, B->A), over K(K-1)/2
    pairs. This is the vector the Spearman correlations of §IV-B use."""
    return np.array([max(Tb[i, j], Tb[j, i])
                     for i in range(K) for j in range(i + 1, K)])


# ------------------------------------------------------------------
# Statistics (paper §IV-B)
# ------------------------------------------------------------------
def within_pairs():
    """Nine within-language cross-method comparisons."""
    return [(l, METHODS[a], METHODS[b])
            for l in LANGS for a in range(3) for b in range(a + 1, 3)]


def cross_pairs():
    """Nine cross-lingual same-method comparisons."""
    return [(m, LANGS[a], LANGS[b])
            for m in METHODS for a in range(3) for b in range(a + 1, 3)]


def delta_rho_stats(within, cross, B=B_BOOT, seed=SEED):
    """Percentile bootstrap CI and label-permutation p for
    delta_rho = mean(within) - mean(cross)."""
    within, cross = np.asarray(within), np.asarray(cross)
    d = within.mean() - cross.mean()
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(within, len(within), replace=True).mean()
                     - rng.choice(cross, len(cross), replace=True).mean()
                     for _ in range(B)])
    ci = np.percentile(boot, [2.5, 97.5])
    pool, nw = np.concatenate([within, cross]), len(within)
    null = np.empty(B)
    for i in range(B):
        perm = rng.permutation(pool)
        null[i] = perm[:nw].mean() - perm[nw:].mean()
    p_perm = (np.sum(null >= d) + 1) / (B + 1)
    return dict(delta_rho=float(d), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                boot_se=float(boot.std()), perm_p=float(p_perm),
                boot=boot, null=null)


def newman_Q(W, communities):
    """Newman modularity of a partition on a weighted undirected graph."""
    W = (W + W.T) / 2.0
    W = W.copy()
    np.fill_diagonal(W, 0)
    m2 = W.sum()
    if m2 == 0:
        return 0.0
    k = W.sum(1)
    Q = 0.0
    for i in range(len(W)):
        for j in range(len(W)):
            if communities[i] == communities[j]:
                Q += W[i, j] - k[i] * k[j] / m2
    return Q / m2


def newman_Q_permtest(W, communities, B=B_BOOT, seed=SEED):
    """Label-permutation null preserving community sizes. §IV-D shows the
    null is strongly non-zero, so Q must not be read without it."""
    rng = np.random.default_rng(seed)
    q_obs = newman_Q(W, communities)
    null = np.array([newman_Q(W, list(rng.permutation(communities)))
                     for _ in range(B)])
    p_lower = (np.sum(null <= q_obs) + 1) / (B + 1)
    return dict(Q=float(q_obs), null_mean=float(null.mean()),
                null_sd=float(null.std()), p_lower=float(p_lower))


# ------------------------------------------------------------------
# Plot style (IEEE single/double column, drawn at true print size)
# ------------------------------------------------------------------
COL1 = 3.5      # IEEE single column, inches
COL2 = 7.16     # IEEE \textwidth, inches

BLUE_D, BLUE_L = '#2166AC', '#92C5DE'
RED_D,  RED_L  = '#D6604D', '#F4A582'
SB_BLUE, SB_ORNG, SB_PURP = '#4C72B0', '#DD8452', '#8172B3'

PLOT_RC = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'figure.dpi': 110, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
}
