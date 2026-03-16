
import os
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import product
import warnings

warnings.filterwarnings("ignore")


# ===========================
# === 1. DATI & CONFIGURAZIONE
# ===========================
symbols = {'SPY': 'SPY', 'GLD': 'GLD', 'VIX': '^VIX'}
csv_file = "portfolio_data.csv"

if os.path.exists(csv_file):
    print("✅ Caricamento dati da file locale...")
    df_raw = pd.read_csv(csv_file, index_col=0, parse_dates=True, header=[0, 1])
else:
    print("⬇️ Download dati da Yahoo Finance...")
    
    tickers_list = list(symbols.values())
    df_raw = yf.download(tickers_list, period="max", interval="1d", group_by='ticker', auto_adjust=True)
    df_raw.to_csv(csv_file)

# ===========================
# === 2. PREPROCESSING & FEATURES 
# ===========================
print("=== Elaborazione Dati Mensili ===")

data_frames = []

for name, ticker in symbols.items():
    
    try:
        
        if ticker in df_raw.columns.levels[0]:
            daily_data = df_raw[ticker]['Close']
        else:
            
            daily_data = df_raw['Close']
    except KeyError:
        print(f"⚠️ Dati mancanti per {name}")
        continue

    #Calcolo Volatilità PRIMA del resample
    # Questo aiuta l'HMM a capire se il mese è stato "nervoso"
    daily_rets = np.log(daily_data / daily_data.shift(1))
    
    # Resample Mensile (Prezzo Fine Mese e Volatilità del Mese)
    monthly_price = daily_data.resample('ME').last() 
    monthly_vol = daily_rets.resample('ME').std() * np.sqrt(21) # Annualizzata mensile
    
    df_temp = pd.DataFrame({
        f'{name}_Price': monthly_price,
        f'{name}_Vol': monthly_vol
    })
    data_frames.append(df_temp)

df_main = pd.concat(data_frames, axis=1)

# Gestione VIX (Se presente, uso la sua media mensile come feature extra)
if 'VIX_Price' in df_main.columns:
   
    df_main['VIX_Price'] = df_main['VIX_Price'].expanding().median().fillna(20.0)
else:
    df_main['VIX_Price'] = 20.0

# --- Calcolo Returns Mensili ---
df_main['SPY_Ret'] = np.log(df_main['SPY_Price'] / df_main['SPY_Price'].shift(1))
df_main['GLD_Ret'] = np.log(df_main['GLD_Price'] / df_main['GLD_Price'].shift(1))

# Pulizia
df_main.dropna(inplace=True)

print(f"✅ Dati pronti: {len(df_main)} mesi.")
print("Feature disponibili per HMM: SPY_Ret, GLD_Ret, SPY_Vol, GLD_Vol")

# ===========================
# === 3. MOTORE WALK-FORWARD
# ===========================
from itertools import product
from sklearn.metrics import confusion_matrix

# Parametri
param_grid = list(product(
    [2, 3],       # Stati HMM
    [3, 6],       # Finestra Volatilità
    [3, 6, 12]    # Finestra Momentum
))

TRAIN_WINDOW = 48   # 4 Anni di training
REBALANCE_MONTHS = 12 # Ricalibrazione annuale
COST_BPS = 0.0010     # Costi transazione

oos_results = []
current_weight_gld = 0.0 # Posizione iniziale (Full Equity)

print("\n=== Inizio Walk-Forward (Logica Shiftata - Ottimizzazione sulla Log-Likelihood ===")

# Loop temporale
for t in range(TRAIN_WINDOW, len(df_main), REBALANCE_MONTHS):
    # Definizione indici temporali
    train_end_idx = t
    train_start_idx = t - TRAIN_WINDOW
    test_end_idx = min(t + REBALANCE_MONTHS, len(df_main))
    
    if train_end_idx >= len(df_main): break
    
    # Dati In-Sample (IS)
    is_data = df_main.iloc[train_start_idx:train_end_idx].copy()
    
    # --- 1. OTTIMIZZAZIONE PARAMETRI (Grid Search su Log-Likelihood) ---
    best_metric = -np.inf 
    best_params = (2, 6, 6) # Default
    
    for n_comp, vol_roll, mom_win in param_grid:
        try:
            # Prep features IS
            temp = is_data.copy()
            temp['Vol'] = temp['SPY_Ret'].rolling(vol_roll).std()
            temp.dropna(inplace=True)
            
            if len(temp) < 24: continue

            # HMM Fit
            X = temp[['SPY_Ret', 'GLD_Ret', 'Vol', 'VIX_Price']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = GMMHMM(n_components=n_comp, covariance_type='full', n_iter=100, random_state=42)
            model.fit(X_scaled)
            
            #Usiamo la Log-Likelihood: quanto bene il modello fitta i dati.
            # Più è alto (meno negativo), meglio è.
            current_score = model.score(X_scaled)
            
            if current_score > best_metric:
                best_metric = current_score
                best_params = (n_comp, vol_roll, mom_win)
                
        except Exception as e:
            continue

    # --- 2. APPLICAZIONE OUT-OF-SAMPLE (OOS) ---
    n_c, v_r, m_w = best_params
    
    # Preparo i dati per il periodo di test + buffer
    buffer = max(v_r, m_w) + 1
    extended_slice = df_main.iloc[train_end_idx - buffer : test_end_idx].copy()
    
    # Calcolo Indicatori definitivi
    extended_slice['Vol'] = extended_slice['SPY_Ret'].rolling(v_r).std()
    extended_slice['Mom'] = extended_slice['SPY_Price'].pct_change(m_w)
    
    oos_data = extended_slice.iloc[buffer:].copy()
    
    if oos_data.empty: continue

    # Retrain modello finale su tutto il periodo IS
    final_train = is_data.copy()
    final_train['Vol'] = final_train['SPY_Ret'].rolling(v_r).std()
    final_train.dropna(inplace=True)
    
    X_train = final_train[['SPY_Ret', 'GLD_Ret', 'Vol', 'VIX_Price']].values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    
    final_model = GMMHMM(n_components=n_c, covariance_type='full', n_iter=200, random_state=42)
    final_model.fit(X_train_sc)
    
    # Identifica Crash State (media SPY più bassa)
    posteriors = final_model.predict_proba(X_train_sc)
    state_means = [np.average(final_train['SPY_Ret'], weights=posteriors[:, s]) for s in range(n_c)]
    crash_state = np.argmin(state_means)
    
    # --- TRADING LOOP ---
    X_oos = oos_data[['SPY_Ret', 'GLD_Ret', 'Vol', 'VIX_Price']].values
    X_oos_sc = scaler.transform(X_oos)
    
    oos_states = final_model.predict(X_oos_sc)
    
    for i in range(len(oos_data)):
        today_state = oos_states[i]
        today_mom = oos_data['Mom'].iloc[i]
        
        # Logica di Allocazione
        target_gld = 0.0
        if today_state == crash_state:
            target_gld = 1.0 # Difesa Totale
        elif today_mom < 0:
            target_gld = 0.6 # Difesa Parziale
        else:
            target_gld = 0.0 # Attacco (SPY)
            
        # Calcolo Return
        r_spy = oos_data['SPY_Ret'].iloc[i]
        r_gld = oos_data['GLD_Ret'].iloc[i]
        
        port_ret = (1 - current_weight_gld) * r_spy + current_weight_gld * r_gld
        
        # Costi
        cost = 0.0
        if abs(target_gld - current_weight_gld) > 0.1:
            cost = COST_BPS
            
        net_ret = port_ret - cost
        current_weight_gld = target_gld
        
        oos_results.append({
            'Date': oos_data.index[i],
            'Strategy': net_ret,
            'SPY_BuyHold': r_spy,
            'GLD_Weight': current_weight_gld, 
            'State': today_state
        })

print(f"✅ Walk-Forward completato. {len(oos_results)} mesi simulati.")

# ===========================
# ===========================
# === 4. ANALISI RISULTATI 
# ===========================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

if not oos_results:
    print("❌ Nessun risultato OOS generato.")
else:
    res_df = pd.DataFrame(oos_results).set_index('Date')
    
   
    if 'GLD' not in res_df.columns:
       
        common_idx = res_df.index.intersection(df_main.index)
        res_df.loc[common_idx, 'GLD'] = df_main.loc[common_idx, 'GLD_Ret']
        # Se c'è qualche NaN residuo (date non matchate), riempio con 0
        res_df['GLD'].fillna(0, inplace=True)

    # 1. Calcolo Equity Curves (Base 100)
    # np.exp della cumsum dei log returns ci dà il fattore di crescita
    res_df['Cum_Strat'] = 100 * res_df['Strategy'].cumsum().apply(np.exp)
    res_df['Cum_SPY'] = 100 * res_df['SPY_BuyHold'].cumsum().apply(np.exp)
    res_df['Cum_GLD'] = 100 * res_df['GLD'].cumsum().apply(np.exp) # Utile per confronto
    
    # Benchmark 60/40 (60% SPY, 40% GLD) - Ribilanciamento continuo approssimato
    res_df['Bench_6040_Ret'] = 0.6 * res_df['SPY_BuyHold'] + 0.4 * res_df['GLD']
    res_df['Cum_Bench'] = 100 * res_df['Bench_6040_Ret'].cumsum().apply(np.exp)
    
    # --- CALCOLO DRAWDOWN ---
    # Drawdown = (Prezzo Attuale / Massimo Storico Precedente) - 1
    def calc_dd(series):
        return (series / series.cummax()) - 1
    
    res_df['DD_Strat'] = calc_dd(res_df['Cum_Strat'])
    res_df['DD_SPY'] = calc_dd(res_df['Cum_SPY'])
    res_df['DD_Bench'] = calc_dd(res_df['Cum_Bench'])

    # --- STATISTICHE AVANZATE ---
    def get_advanced_stats(returns_log, drawdown_series):
        # CAGR (Rendimento Composto Annuo)
        total_months = len(returns_log)
        total_ret = np.exp(returns_log.sum()) - 1
        cagr = (1 + total_ret) ** (12 / total_months) - 1 #Tasso di crescita annuo composto
        
        # Volatilità
        ann_vol = returns_log.std() * np.sqrt(12)
        
        # Sharpe (Risk Free = 0)
        sharpe = cagr / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown
        mdd = drawdown_series.min()
        
        # Calmar Ratio
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        
        return {'CAGR': cagr, 'Vol': ann_vol, 'Sharpe': sharpe, 'MaxDD': mdd, 'Calmar': calmar}

    stats_strat = get_advanced_stats(res_df['Strategy'], res_df['DD_Strat'])
    stats_spy = get_advanced_stats(res_df['SPY_BuyHold'], res_df['DD_SPY'])
    stats_bench = get_advanced_stats(res_df['Bench_6040_Ret'], res_df['DD_Bench'])

    print("\n" + "="*65)
    print(f"=== REPORT FINALE (Walk-Forward: {len(res_df)} mesi) ===")
    print("="*65)
    print(f"{'Metrica':<15} | {'STRATEGIA':<12} | {'S&P 500':<12} | {'60/40 Bench':<12}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {stats_strat['CAGR']:>11.1%} | {stats_spy['CAGR']:>11.1%} | {stats_bench['CAGR']:>11.1%}")
    print(f"{'Volatilità':<15} | {stats_strat['Vol']:>11.1%} | {stats_spy['Vol']:>11.1%} | {stats_bench['Vol']:>11.1%}")
    print(f"{'Max Drawdown':<15} | {stats_strat['MaxDD']:>11.1%} | {stats_spy['MaxDD']:>11.1%} | {stats_bench['MaxDD']:>11.1%}")
    print(f"{'Sharpe Ratio':<15} | {stats_strat['Sharpe']:>11.2f} | {stats_spy['Sharpe']:>11.2f} | {stats_bench['Sharpe']:>11.2f}")
    print(f"{'Calmar Ratio':<15} | {stats_strat['Calmar']:>11.2f} | {stats_spy['Calmar']:>11.2f} | {stats_bench['Calmar']:>11.2f}")
    print("-" * 65)

    # --- GRAFICI (3 Pannelli) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. Equity Curve
    ax1.plot(res_df.index, res_df['Cum_Strat'], label='HMM Strategy', color='#1f77b4', linewidth=2.5)
    ax1.plot(res_df.index, res_df['Cum_SPY'], label='S&P 500', color='gray', alpha=0.5, linestyle='--')
    ax1.plot(res_df.index, res_df['Cum_Bench'], label='60/40 Benchmark', color='orange', alpha=0.6, linestyle='-.')
    ax1.set_title(f"Crescita del Capitale (Start=100)")
    ax1.set_ylabel("Capitale (Log Scale)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_yscale('log')

    # 2. Allocazione (Area Chart)
    ax2.fill_between(res_df.index, 0, res_df['GLD_Weight'], color='gold', alpha=0.5, label='Allocazione ORO')
    ax2.plot(res_df.index, res_df['GLD_Weight'], color='goldenrod', linewidth=1)
    ax2.set_ylabel("Peso Oro (0-1)")
    ax2.set_title("Regime Detection (Giallo = Risk Off/Crash Mode)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Underwater Plot
    ax3.fill_between(res_df.index, 0, res_df['DD_Strat'], color='red', alpha=0.3, label='Drawdown Strategia')
    ax3.plot(res_df.index, res_df['DD_SPY'], color='gray', linestyle=':', alpha=0.5, label='Drawdown SPY')
    ax3.set_ylabel("Perdita dai Massimi")
    ax3.set_title(f"Sofferenza (MaxDD: {stats_strat['MaxDD']:.1%})")
    ax3.legend(loc="lower left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ===========================
# === 5. TEST STATISTICI AVANZATI (VALIDAZIONE ISTITUZIONALE) ===
# ===========================
from scipy.stats import skew, kurtosis, norm

def advanced_stats_report(returns, benchmark_sr=0, confidence_level=0.99):
    """
    Calcolo PSR (Probabilistic Sharpe Ratio) e MinTRL (Minimum Track Record Length).
    Fonte: Bailey, D. H., & Lopez de Prado, M. (2012).
    """
    # 1. Metriche della distribuzione
    n = len(returns)
    sr_est = returns.mean() / returns.std() * np.sqrt(12) # Annualizzato
    sk = skew(returns)
    kr = kurtosis(returns) # Fisher (normal = 0)
    
    # 2. Probabilistic Sharpe Ratio (PSR)
    # Penalizza lo Sharpe se c'è asimmetria negativa o code grasse
    sigma_sr = np.sqrt((1 - sk * sr_est + (kr - 1) / 4 * sr_est**2) / (n - 1))
    z = (sr_est - benchmark_sr) / sigma_sr
    psr = norm.cdf(z)
    
    # 3. Minimum Track Record Length (MinTRL)
    # Quanti anni servono per essere sicuri al 99% che Sharpe > benchmark_sr?
    if sr_est > benchmark_sr:
        alpha = 1 - confidence_level
        z_alpha = norm.ppf(1 - alpha)
        
        # Formula MinTRL (in anni)
        min_trl = 12 * (z_alpha * sigma_sr * np.sqrt(n-1) / (sr_est - benchmark_sr))**2 / 12
        # Nota: La formula originale calcola N campioni, noi dividiamo per 12 per avere anni
        # Semplificazione numerica diretta:
        numerator = 1 - sk * sr_est + (kr - 1) / 4 * sr_est**2
        denominator = (sr_est - benchmark_sr)**2
        min_trl_years = 1 * (z_alpha**2 * numerator) / denominator
    else:
        min_trl_years = np.inf 
        
    return {
        'Sharpe': sr_est,
        'Skew': sk,
        'Kurtosis': kr,
        'PSR': psr,
        'MinTRL': min_trl_years
    }

# Analisi Strategia vs SPY
stats_strat = advanced_stats_report(res_df['Strategy'], benchmark_sr=0) # Test vs 0 (Profitto Assoluto)
stats_alpha = advanced_stats_report(res_df['Strategy'] - res_df['SPY_BuyHold'], benchmark_sr=0) # Test vs Market (Alpha)

print("\n" + "="*60)
print(" 🔬 REPORT DI VALIDAZIONE STATISTICA (LOPEZ DE PRADO)")
print("="*60)

print(f"1. ANALISI DISTRIBUZIONE (Rischio Nascosto)")
print(f"   Sharpe Annualizzato:  {stats_strat['Sharpe']:.2f}")
print(f"   Skewness (Asimmetria): {stats_strat['Skew']:.2f}")
print(f"      -> {'✅ Positiva (Profitti esplosivi)' if stats_strat['Skew']>0 else '⚠️ Negativa (Rischio Crash rapido)'}")
print(f"   Kurtosis (Code Grasse): {stats_strat['Kurtosis']:.2f}")
print(f"      -> {'✅ Normale' if stats_strat['Kurtosis']<3 else '⚠️ Alta (Eventi estremi frequenti)'}")

print("-" * 60)
print(f"2. PROBABILISTIC SHARPE RATIO (PSR)")
print(f"   Probabilità che la strategia sia profittevole (SR > 0): {stats_strat['PSR']:.2%}")
if stats_strat['PSR'] > 0.95:
    print("   ✅ RISULTATO: La strategia è statisticamente robusta (>95%).")
else:
    print("   ⚠️ RISULTATO: Non c'è abbastanza certezza statistica.")

print("-" * 60)
print(f"3. MINIMUM TRACK RECORD LENGTH (MinTRL)")
print(f"   Anni di storico necessari per validare la strategia al 99%: {stats_strat['MinTRL']:.1f} anni")
print(f"   Anni di storico simulati nel Walk-Forward: {len(res_df)/12:.1f} anni")

if len(res_df)/12 > stats_strat['MinTRL']:
    print("   ✅ VALIDATO: Abbiamo dati sufficienti per fidarci del risultato.")
else:
    print("   ⚠️ ATTENZIONE: Il backtest è troppo breve rispetto alla volatilità della strategia.")

print("-" * 60)
print(f"4. TEST DI ALPHA (Battono il mercato?)")
print(f"   Probabilità che la Strategia sia meglio di SPY: {stats_alpha['PSR']:.2%}")
print("="*60)

# ===========================
# === 6. PREVISIONE OPERATIVA (MODE: LOG-LIKELIHOOD) ===
# ===========================
print("\n" + "="*50)
print(" 🔮 PREVISIONE OPERATIVA PER IL MESE SUCCESSIVO ")
print("="*50)

# 1. Recupero dati recenti (Ultimi 5 anni per calibrazione)
latest_data = df_main.iloc[-TRAIN_WINDOW:].copy()

# 2. Ottimizzazione Parametri (Coerenza con Blocco 3)
# Cerchiamo il modello che descrive meglio la statistica attuale del mercato.
best_metric_now = -np.inf 
best_params_now = (2, 6, 6) # Default conservativo

print("⚙️ Calcolo parametri ottimali attuali (basati sulla Likelihood)...")

for n_comp, vol_roll, mom_win in param_grid:
    try:
        temp = latest_data.copy()
        # Calcolo indicatori
        temp['Vol_SPY'] = temp['SPY_Ret'].rolling(vol_roll).std()
        temp.dropna(inplace=True)
        
        if len(temp) < 24: continue
        
        X = temp[['SPY_Ret', 'GLD_Ret', 'Vol_SPY', 'VIX_Price']].values
        scaler_opt = StandardScaler()
        X_scaled = scaler_opt.fit_transform(X)
        
        model = GMMHMM(n_components=n_comp, covariance_type='full', n_iter=100, random_state=42)
        model.fit(X_scaled)
        
        current_score = model.score(X_scaled)
        
        if current_score > best_metric_now:
            best_metric_now = current_score
            best_params_now = (n_comp, vol_roll, mom_win)
            
    except: continue

n_comp_now, vol_roll_now, mom_win_now = best_params_now
print(f"✅ Parametri Vincenti: Stati={n_comp_now}, Vol_Roll={vol_roll_now}m, Momentum={mom_win_now}m")
print(f"   (Log-Likelihood Score: {best_metric_now:.2f})")

# 3. Addestramento Finale (Con i parametri vincenti)
# Ricalcoliamo le feature finali sul dataset
latest_data['Vol_SPY'] = latest_data['SPY_Ret'].rolling(vol_roll_now).std()
latest_data['SPY_Mom'] = latest_data['SPY_Price'].pct_change(mom_win_now) 
latest_data.dropna(inplace=True)

X_last = latest_data[['SPY_Ret', 'GLD_Ret', 'Vol_SPY', 'VIX_Price']].values
scaler_final = StandardScaler()
X_last_scaled = scaler_final.fit_transform(X_last)

final_model_now = GMMHMM(n_components=n_comp_now, covariance_type='full', n_iter=200, random_state=42)
final_model_now.fit(X_last_scaled)

# 4. Identificazione Stati
posteriors = final_model_now.predict_proba(X_last_scaled)
# Calcolo rendimento medio SPY per ogni stato per trovare il peggiore
state_means_spy = [np.average(latest_data['SPY_Ret'], weights=posteriors[:, s]) for s in range(n_comp_now)]
crash_state_now = np.argmin(state_means_spy)
# Volatilità media per stato (per info)
state_means_vol = [np.average(latest_data['Vol_SPY'], weights=posteriors[:, s]) for s in range(n_comp_now)]

# 5. PREVISIONE
last_features = X_last_scaled[-1].reshape(1, -1)
current_state = final_model_now.predict(last_features)[0]
current_probs = final_model_now.predict_proba(last_features)[0]
prob_crash = current_probs[crash_state_now]
current_momentum = latest_data['SPY_Mom'].iloc[-1]

# 6. Logica Decisionale Finale
rec_gld = 0.0
msg = ""
icon = ""

# Definiamo soglie
if current_state == crash_state_now:
    rec_gld = 1.0
    msg = f"HMM RILEVA CRASH/ALTA VOLATILITÀ (Prob: {prob_crash:.1%})"
    icon = "🔴 RISK OFF"
elif current_momentum < 0:
    rec_gld = 0.6
    msg = f"HMM OK, MA MOMENTUM NEGATIVO ({current_momentum:.1%})"
    icon = "🟡 PRUDENZA"
else:
    rec_gld = 0.0
    msg = f"HMM OK & MOMENTUM POSITIVO ({current_momentum:.1%})"
    icon = "🟢 RISK ON"

rec_spy = 1.0 - rec_gld

# OUTPUT PER L'UTENTE
print("\n" + "-"*40)
print(f"📅 DATA ANALISI: {latest_data.index[-1].date()}")
print("-" * 40)
print(f"Stato HMM Attuale: {current_state} (Su {n_comp_now} possibili)")
print(f" -> Stato Crash è il n.{crash_state_now} (Volatilità media stato: {state_means_vol[crash_state_now]*np.sqrt(12):.1%})")
print(f" -> Probabilità di essere nel Crash State: {prob_crash:.1%}")
print("-" * 40)
print("⚖️  PORTAFOGLIO SUGGERITO:")
print(f"   {icon} ")
print(f"   SPY (Azionario): {rec_spy:.0%}")
print(f"   GLD (Oro):       {rec_gld:.0%}")
print(f"   Motivo:          {msg}")
print("-" * 40 + "\n")