"""QSWAT+ + ML stacking workflow for Upper Ganale streamflow.

Historical input combines the user-provided daily observed and QSWAT+ flow files with the five-station
daily climate table. Evaluation uses the available overlapping 1988–2011 period.
"""
from pathlib import Path
import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

PROJECT_DIR = Path(os.environ.get('HYBRID_PROJECT_DIR', Path(__file__).resolve().parent)).expanduser()
DATA_DIR = Path(os.environ.get('HYBRID_DATA_DIR', PROJECT_DIR / 'data')).expanduser()
OBS_FILE = Path(os.environ.get('Q_OBSERVATION_FILE', DATA_DIR / 'Q_observation_historical.csv')).expanduser()
QSWAT_FILE = Path(os.environ.get('Q_SIMULATION_FILE', DATA_DIR / 'Q_simulation_historical.csv')).expanduser()
STATION_CLIMATE_FILE = Path(os.environ.get(
    'STATION_CLIMATE_FILE', DATA_DIR / 'combined_climate_with_station_names.csv')).expanduser()
CMIP6_DIR = Path(os.environ.get('CMIP6_DATA_DIR', DATA_DIR / 'CMIP6')).expanduser()
CMIP6_RANKING_FILE = Path(os.environ.get(
    'CMIP6_RANKING_FILE', DATA_DIR / 'combined_cri_scores.csv')).expanduser()
QSWAT_FUTURE_DIR = Path(os.environ.get('QSWAT_FUTURE_DIR', DATA_DIR / 'qswat_scenarios')).expanduser()
OUT_DIR = PROJECT_DIR / 'hybrid_outputs'
OUT_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42
FEATURES = ['Q_Simulated','avg_pcp','avg_tmax','avg_tmin','avg_slr','avg_wnd','avg_hmd',
            'observed_flow_lag1','simulated_flow_lag1','avg_pcp_lag1','observed_flow_lag7',
            'avg_pcp_lag7','year','month','day_of_year','day_of_week','week_of_year',
            'rolling_avg_pcp_7d','rolling_avg_observed_flow_7d']

def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ','_') for c in df.columns]
    aliases = {'datetime':'date','time':'date','observed':'observed_flow','q_observed':'observed_flow',
               'qswat':'Q_Simulated','qswat_flow':'Q_Simulated','q_simulated':'Q_Simulated',
               'p_mm':'avg_pcp','precipitation':'avg_pcp','pr':'avg_pcp','tmax_c':'avg_tmax',
               'tmax':'avg_tmax','tasmax':'avg_tmax','tmin_c':'avg_tmin','tmin':'avg_tmin','tasmin':'avg_tmin'}
    return df.rename(columns={c:aliases[c] for c in df.columns if c in aliases})

def features(df, historical=False):
    d = norm(df)
    if 'date' not in d: raise ValueError(f'No date column. Found: {list(d.columns)}')
    d['date'] = pd.to_datetime(d.date, errors='coerce')
    d = d.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    for c,v in [('avg_slr',18.),('avg_wnd',2.),('avg_hmd',.6)]:
        if c not in d: d[c] = v
    for c in ['Q_Simulated','avg_pcp','avg_tmax','avg_tmin']:
        if c not in d: raise ValueError(f'Missing required column {c}')
    d['year']=d.date.dt.year; d['month']=d.date.dt.month; d['day_of_year']=d.date.dt.dayofyear
    d['day_of_week']=d.date.dt.dayofweek; d['week_of_year']=d.date.dt.isocalendar().week.astype(int)
    d['avg_pcp_lag1']=d.avg_pcp.shift(1); d['avg_pcp_lag7']=d.avg_pcp.shift(7)
    d['rolling_avg_pcp_7d']=d.avg_pcp.rolling(7,min_periods=7).mean()
    d['simulated_flow_lag1']=d.Q_Simulated.shift(1)
    if historical:
        if 'observed_flow' not in d: raise ValueError('Historical CSV needs observed_flow')
        d['observed_flow_lag1']=d.observed_flow.shift(1); d['observed_flow_lag7']=d.observed_flow.shift(7)
        d['rolling_avg_observed_flow_7d']=d.observed_flow.rolling(7,min_periods=7).mean()
    return d

def metrics(obs,pred):
    obs=np.asarray(obs,float); pred=np.asarray(pred,float); den=np.sum((obs-obs.mean())**2); total=np.sum(obs)
    return {'R2':r2_score(obs,pred),'NSE':1-np.sum((obs-pred)**2)/den if den else np.nan,
            'RMSE':float(np.sqrt(mean_squared_error(obs,pred))),
            'MAE':float(mean_absolute_error(obs,pred)),
            'PBIAS_percent':100*np.sum(obs-pred)/total if total else np.nan}

def fit_bma_weights(observed,predictions,max_iter=1000,tol=1e-8):
    """Estimate Gaussian-mixture BMA weights from calibration-only OOF predictions."""
    y=np.asarray(observed,dtype=float).reshape(-1,1)
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or p.shape[0]!=len(y) or not np.isfinite(p).all():
        raise ValueError('BMA needs a complete finite out-of-fold calibration prediction matrix.')
    k=p.shape[1]; weights=np.full(k,1.0/k)
    variances=np.maximum(np.mean((y-p)**2,axis=0),1e-6)
    for _ in range(max_iter):
        log_prob=np.log(np.maximum(weights,1e-12))[None,:]-.5*(np.log(2*np.pi*variances)[None,:]+(y-p)**2/variances[None,:])
        log_prob-=log_prob.max(axis=1,keepdims=True)
        responsibility=np.exp(log_prob); responsibility/=responsibility.sum(axis=1,keepdims=True)
        new_weights=np.maximum(responsibility.mean(axis=0),1e-12); new_weights/=new_weights.sum()
        new_variances=np.maximum((responsibility*(y-p)**2).sum(axis=0)/np.maximum(responsibility.sum(axis=0),1e-12),1e-6)
        if np.max(np.abs(new_weights-weights))<tol and np.max(np.abs(new_variances-variances))<tol:
            weights,variances=new_weights,new_variances; break
        weights,variances=new_weights,new_variances
    return weights,variances

def load_historical():
    for p in (OBS_FILE,QSWAT_FILE,STATION_CLIMATE_FILE):
        if not p.exists(): raise FileNotFoundError(f'Required historical input missing: {p}')
    obs=pd.read_csv(OBS_FILE); sim=pd.read_csv(QSWAT_FILE); clim=pd.read_csv(STATION_CLIMATE_FILE)
    obs.columns=[str(c).strip().lower() for c in obs.columns]
    sim.columns=[str(c).strip().lower() for c in sim.columns]
    clim.columns=[str(c).strip().lower() for c in clim.columns]
    obs=obs.rename(columns={'observed-flow':'observed_flow','observed_flow':'observed_flow'})
    for d in (obs,sim,clim):
        if 'date' not in d: raise ValueError(f'Expected Date column in input; columns found: {list(d.columns)}')
        d['date']=pd.to_datetime(d.date,format='mixed',dayfirst=True,errors='coerce')
    if 'q_simulated' in sim: sim=sim.rename(columns={'q_simulated':'Q_Simulated'})
    clim_daily=clim.groupby('date',as_index=False).agg(
        avg_pcp=('pcp','mean'),avg_tmax=('tmax','mean'),avg_tmin=('tmin','mean'),
        avg_slr=('slr','mean'),avg_wnd=('wnd','mean'),avg_hmd=('hmd','mean'))
    merged=(obs[['date','observed_flow']].merge(sim[['date','Q_Simulated']],on='date',how='inner')
            .merge(clim_daily,on='date',how='inner').sort_values('date').reset_index(drop=True))
    merged.to_csv(OUT_DIR/'merged_historical_training_input.csv',index=False)
    return merged

def make_lstm():
    model=Sequential([Input(shape=(1,len(FEATURES))),LSTM(50,activation='relu'),Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    return model

def predict_lstm_single_step(model,scaled_row):
    """Fast NumPy inference equivalent for this one-timestep LSTM architecture."""
    cell=next(layer for layer in model.layers if layer.__class__.__name__=='LSTM')
    dense=[layer for layer in model.layers if layer.__class__.__name__=='Dense']
    kernel,_,bias=cell.get_weights()
    x=np.asarray(scaled_row,dtype=np.float32).reshape(1,-1)
    z=x@kernel+bias
    n=cell.units
    sigmoid=lambda a:1.0/(1.0+np.exp(-np.clip(a,-60.0,60.0)))
    # LSTM activation is ReLU in this model; its state is reset for each one-step input.
    h=sigmoid(z[:,3*n:])*np.maximum(0.0,sigmoid(z[:,:n])*np.maximum(0.0,z[:,2*n:3*n]))
    for layer in dense:
        weights,offset=layer.get_weights(); h=h@weights+offset
        if layer.activation.__name__=='relu': h=np.maximum(0.0,h)
    return float(np.asarray(h).ravel()[0])

def run():
    set_random_seed(SEED)
    h=features(load_historical(),historical=True)
    h=h[h.date.between('1988-01-01','2011-12-31')].dropna(subset=FEATURES+['observed_flow']).reset_index(drop=True)
    if len(h)<100: raise ValueError(f'Only {len(h)} usable daily rows in 1988–2011.')
    calibration=h[h.date.between('1988-01-01','2002-12-31')].reset_index(drop=True)
    validation=h[h.date.between('2003-01-01','2011-12-31')].reset_index(drop=True)
    if calibration.empty or validation.empty: raise ValueError('Chronological calibration/validation period is empty.')
    Xtr=calibration[FEATURES].astype(float); ytr=calibration.observed_flow.astype(float)
    Xv=validation[FEATURES].astype(float); yv=validation.observed_flow.astype(float)
    cv=TimeSeriesSplit(n_splits=3)
    # All tuning and time-ordered CV folds are confined to 1988-2002.
    specs={
      'rf':(RandomForestRegressor(n_jobs=1,random_state=SEED),
            {'n_estimators':[120,220,320],'max_depth':[None,12,24],'min_samples_leaf':[1,2,4],'max_features':['sqrt',.8,1.0]}),
      'xgb':(XGBRegressor(objective='reg:squarederror',n_jobs=1,random_state=SEED,verbosity=0),
             {'n_estimators':[120,220,320],'max_depth':[3,5,7],'learning_rate':[.025,.05,.1],'subsample':[.8,1.0],'colsample_bytree':[.8,1.0]}),
      'lgbm':(LGBMRegressor(n_jobs=1,verbosity=-1,random_state=SEED),
              {'n_estimators':[120,220,320],'num_leaves':[15,31,47],'learning_rate':[.025,.05,.1],'min_child_samples':[10,20,30]})}
    searches={}
    for name,(est,params) in specs.items():
        search=RandomizedSearchCV(est,params,n_iter=4,scoring='r2',cv=cv,random_state=SEED,n_jobs=1,refit=True)
        search.fit(Xtr,ytr); searches[name]=search
        print(f'{name} 3-fold best CV R²={search.best_score_:.4f}; params={search.best_params_}')
    rf,xgb,lgbm=(searches[k].best_estimator_ for k in ('rf','xgb','lgbm'))
    # Expanding-window out-of-fold base predictions avoid in-sample stacking features.
    oof=pd.DataFrame(index=np.arange(len(Xtr)))
    oof['QSWAT_Pred']=Xtr.Q_Simulated.to_numpy()
    base_cv_metrics={}
    for name,est in [('RF_Pred',rf),('XGB_Pred',xgb),('LightGBM_Pred',lgbm)]:
        oof[name]=np.nan; base_cv_metrics[name]=[]
        for fit_idx,hold_idx in cv.split(Xtr):
            fold_est=clone(est).fit(Xtr.iloc[fit_idx],ytr.iloc[fit_idx])
            pred=fold_est.predict(Xtr.iloc[hold_idx])
            oof.loc[hold_idx,name]=pred
            base_cv_metrics[name].append(metrics(ytr.iloc[hold_idx],pred))
    oof['LSTM_Pred']=np.nan; lstm_cv_metrics=[]
    for fold,(fit_idx,hold_idx) in enumerate(cv.split(Xtr),start=1):
        fold_scaler=StandardScaler().fit(Xtr.iloc[fit_idx])
        fit_scaled=fold_scaler.transform(Xtr.iloc[fit_idx]).reshape(-1,1,len(FEATURES))
        hold_scaled=fold_scaler.transform(Xtr.iloc[hold_idx]).reshape(-1,1,len(FEATURES))
        fold_model=make_lstm(); fold_model.fit(fit_scaled,ytr.iloc[fit_idx].to_numpy(),epochs=50,batch_size=128,verbose=0,validation_split=.1,
            callbacks=[EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)])
        fold_pred=fold_model.predict(hold_scaled,verbose=0).ravel(); oof.loc[hold_idx,'LSTM_Pred']=fold_pred
        lstm_cv_metrics.append(metrics(ytr.iloc[hold_idx],fold_pred))
    oof_mask=oof[['RF_Pred','XGB_Pred','LightGBM_Pred','LSTM_Pred']].notna().all(axis=1)
    oof_meta=oof.loc[oof_mask].reset_index(drop=True)
    y_meta=ytr.loc[oof_mask].reset_index(drop=True)
    meta=RidgeCV(alphas=[.1,1.,10.,100.],cv=cv).fit(oof_meta,y_meta)
    # Nested time-ordered CV: select Ridge alpha using only each outer fold's past.
    stack_cv_metrics=[]; stack_cv_pred=np.full(len(oof_meta),np.nan)
    for fit_idx,hold_idx in cv.split(oof_meta):
        inner_cv=TimeSeriesSplit(n_splits=3)
        fold_meta=RidgeCV(alphas=[.1,1.,10.,100.],cv=inner_cv).fit(oof_meta.iloc[fit_idx],y_meta.iloc[fit_idx])
        pred=np.maximum(0,fold_meta.predict(oof_meta.iloc[hold_idx]))
        stack_cv_pred[hold_idx]=pred
        stack_cv_metrics.append(metrics(y_meta.iloc[hold_idx],pred))
    scaler=StandardScaler().fit(Xtr); strn=scaler.transform(Xtr); sv=scaler.transform(Xv)
    lstm=make_lstm(); lstm_history=lstm.fit(strn.reshape(-1,1,len(FEATURES)),ytr.to_numpy(),epochs=50,batch_size=128,verbose=0,validation_split=.1,
        callbacks=[EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)])
    rf.fit(Xtr,ytr); xgb.fit(Xtr,ytr); lgbm.fit(Xtr,ytr)
    def base(xx,ss):
        return pd.DataFrame({'QSWAT_Pred':xx.Q_Simulated.to_numpy(),'RF_Pred':rf.predict(xx),
          'XGB_Pred':xgb.predict(xx),'LightGBM_Pred':lgbm.predict(xx),
          'LSTM_Pred':lstm.predict(ss.reshape(-1,1,len(FEATURES)),verbose=0).ravel()})
    # Calibration score uses nested out-of-fold predictions; validation is untouched.
    train_base=base(Xtr,strn); val_base=base(Xv,sv)
    stack_train_pred=np.maximum(0,meta.predict(oof_meta))
    pv=np.maximum(0,meta.predict(val_base))
    bma_models=['QSWAT_Pred','RF_Pred','XGB_Pred','LightGBM_Pred','LSTM_Pred']
    bma_weights,bma_variances=fit_bma_weights(y_meta,oof_meta[bma_models].to_numpy())
    bma_val_pred=np.maximum(0,val_base[bma_models].to_numpy()@bma_weights)
    pd.DataFrame({'Model':bma_models,'Weight':bma_weights,'Residual_variance':bma_variances}).to_csv(OUT_DIR/'bma_weights.csv',index=False)
    def fold_summary(model,fold_metrics):
        return {'Model':model,'R2_mean':np.mean([v['R2'] for v in fold_metrics]),'R2_std':np.std([v['R2'] for v in fold_metrics],ddof=1),
                'NSE_mean':np.mean([v['NSE'] for v in fold_metrics]),'NSE_std':np.std([v['NSE'] for v in fold_metrics],ddof=1),
                'RMSE_mean':np.mean([v['RMSE'] for v in fold_metrics]),'RMSE_std':np.std([v['RMSE'] for v in fold_metrics],ddof=1),
                'MAE_mean':np.mean([v['MAE'] for v in fold_metrics]),'MAE_std':np.std([v['MAE'] for v in fold_metrics],ddof=1)}
    cv_table=pd.DataFrame([fold_summary('Random Forest',base_cv_metrics['RF_Pred']),
      fold_summary('XGBoost',base_cv_metrics['XGB_Pred']),
      fold_summary('LightGBM',base_cv_metrics['LightGBM_Pred']),
      fold_summary('LSTM',lstm_cv_metrics),fold_summary('Stacking Ensemble',stack_cv_metrics)])
    cv_table.to_csv(OUT_DIR/'cross_validation_results.csv',index=False)
    pd.DataFrame([{'Model':k,'Best_CV_R2':v.best_score_,'Best_parameters':str(v.best_params_)} for k,v in searches.items()]).to_csv(OUT_DIR/'hyperparameter_search_results.csv',index=False)
    calibration_cv_mask=np.isfinite(stack_cv_pred)
    calibration_cv_metrics=metrics(y_meta[calibration_cv_mask],stack_cv_pred[calibration_cv_mask])
    report=pd.DataFrame([calibration_cv_metrics,metrics(yv,pv),metrics(yv,bma_val_pred)],index=['Stacking CV (1988-2002)','Stacking Validation (2003-2011)','BMA Validation (2003-2011)'])
    print(f'Calibration period used: {h.date.min().date()} to {h.date.max().date()} ({len(h):,} usable days)')
    print(f'Chronological split: {calibration.date.min().date()} to {calibration.date.max().date()} calibration ({len(Xtr):,} rows); {validation.date.min().date()} to {validation.date.max().date()} validation ({len(Xv):,} rows).')
    print(f'TimeSeriesSplit folds=3, calibration-only; Ridge alpha={meta.alpha_}; LSTM epochs={len(lstm_history.history["loss"])}')
    print('Three-fold expanding-window CV metrics (calibration period only):'); print(cv_table.round(4).to_string(index=False))
    print(report.round(4)); report.to_csv(OUT_DIR/'calibration_validation_metrics.csv')
    performance=[]
    def add_row(model,stage,period,score,remarks):
        performance.append({'Model':model,'Stage':stage,'Period':period,'R2':score['R2'],'NSE':score['NSE'],
          'RMSE':score['RMSE'],'MAE':score['MAE'],'Remarks':remarks})
    def add_cv_row(model,fold_metrics):
        row={'Model':model,'Stage':'Time-series CV','Period':'1988-2002','Remarks':'Mean ± SD across 3 expanding folds'}
        for metric in ['R2','NSE','RMSE','MAE']:
            vals=np.array([m[metric] for m in fold_metrics],float)
            row[metric]=f'{vals.mean():.4f} ± {vals.std(ddof=1):.4f}'
        performance.append(row)
    add_row('QSWAT+','Calibration','1988-2002',metrics(ytr,Xtr.Q_Simulated),'Process-based model')
    add_row('QSWAT+','Validation','2003-2011',metrics(yv,Xv.Q_Simulated),'Independent period')
    model_specs=[('RF','RF_Pred',base_cv_metrics['RF_Pred']),('XGBoost','XGB_Pred',base_cv_metrics['XGB_Pred']),
                 ('LightGBM','LightGBM_Pred',base_cv_metrics['LightGBM_Pred']),('LSTM','LSTM_Pred',lstm_cv_metrics)]
    for label,col,folds in model_specs:
        add_row(label,'Training','1988-2002',metrics(ytr,train_base[col]),'Fitted on calibration period')
        add_cv_row(label,folds)
        add_row(label,'Validation','2003-2011',metrics(yv,val_base[col]),'Independent period')
    add_row('Stacking Ensemble','Training','1988-2002',metrics(y_meta,stack_train_pred),'Ridge on out-of-fold base features')
    add_cv_row('Stacking Ensemble',stack_cv_metrics)
    add_row('Stacking Ensemble','Validation','2003-2011',metrics(yv,pv),'Independent period')
    add_row('BMA','Validation','2003-2011',metrics(yv,bma_val_pred),'Gaussian-mixture weights fit to calibration OOF predictions')
    performance_table=pd.DataFrame(performance)
    performance_table.to_csv(OUT_DIR/'model_performance_table.csv',index=False)
    print('Table X: chronological model performance')
    print(performance_table.to_string(index=False))
    calibration_cv_dates=calibration.date.loc[oof_mask].reset_index(drop=True)
    cv_eval=calibration_cv_mask
    pred_rows=pd.DataFrame({'date':pd.concat([calibration_cv_dates.loc[cv_eval],validation.date],ignore_index=True),
      'partition':['Calibration_CV']*int(cv_eval.sum())+['Validation']*len(validation),
      'observed_flow':pd.concat([y_meta.loc[cv_eval].reset_index(drop=True),yv.reset_index(drop=True)],ignore_index=True),
      'hybrid_predicted_flow':np.concatenate([stack_cv_pred[cv_eval],pv]),
      'bma_predicted_flow':np.concatenate([np.full(int(cv_eval.sum()),np.nan),bma_val_pred])})
    pred_rows.to_csv(OUT_DIR/'chronological_calibration_validation_predictions.csv',index=False)
    validation_preds=pd.DataFrame({'date':validation.date,'observed_flow':yv,'QSWAT_Pred':Xv.Q_Simulated,
      'RF_Pred':val_base.RF_Pred,'XGB_Pred':val_base.XGB_Pred,'LightGBM_Pred':val_base.LightGBM_Pred,
      'LSTM_Pred':val_base.LSTM_Pred,'Stacking_Ensemble':pv,'BMA':bma_val_pred})
    validation_preds.to_csv(OUT_DIR/'model_validation_predictions.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,obs,pred,title in [(axs[0],y_meta[calibration_cv_mask],stack_cv_pred[calibration_cv_mask],'Stacking Calibration CV (1988-2002)'),(axs[1],yv,pv,'Stacking Validation (2003-2011)')]:
        ax.scatter(obs,pred,s=9,alpha=.4); lo=min(obs.min(),pred.min()); hi=max(obs.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],'r--'); z=metrics(obs,pred)
        ax.set(title=f"{title}\nR²={z['R2']:.3f} NSE={z['NSE']:.3f} PBIAS={z['PBIAS_percent']:.2f}%",
               xlabel='Observed streamflow (m³/s)',ylabel='Predicted streamflow (m³/s)'); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(OUT_DIR/'calibration_validation_scatter.png',dpi=180); plt.close(fig)
    joblib.dump({'features':FEATURES,'rf':rf,'xgb':xgb,'lgbm':lgbm,'meta':meta,'scaler':scaler,'seed':SEED,
                 'calibration_period':('1988-01-01','2002-12-31'),'validation_period':('2003-01-01','2011-12-31'),
                 'bma_models':bma_models,'bma_weights':bma_weights,'bma_variances':bma_variances,
                 'selected_hyperparameters':{k:v.best_params_ for k,v in searches.items()},'split_sizes':(len(Xtr),len(Xv))},OUT_DIR/'hybrid_models.joblib')
    lstm.save(OUT_DIR/'lstm_model.keras')
    print('Saved trained models and diagnostics in',OUT_DIR.resolve())
    return h,rf,xgb,lgbm,lstm,meta,scaler

def run_cmip6(h,rf,xgb,lgbm,lstm,meta,scaler):
    ranking=CMIP6_RANKING_FILE
    if not ranking.exists(): raise FileNotFoundError(f'{ranking} missing; need the evaluated CMIP6 model ranking.')
    ranks=pd.read_csv(ranking)
    if not {'Model_Name','Rainfall_CRI'}.issubset(ranks.columns): raise ValueError('Ranking CSV requires Model_Name and Rainfall_CRI columns.')
    selected=ranks.sort_values('Rainfall_CRI',ascending=False).head(6).copy()
    selected[['Model_Name','Rainfall_CRI']].to_csv(
        OUT_DIR/'cmip6_precipitation_model_selection.csv',index=False)
    models=selected.Model_Name.astype(str).tolist()
    print('Six-model precipitation ensemble:',', '.join(models))
    if len(models)<6: raise ValueError('Six precipitation models required. The request names five; provide the sixth in the ranking CSV.')
    def climate(model,scenario):
        p=CMIP6_DIR/model/f'{model}_daily_data.csv'
        if not p.exists(): raise FileNotFoundError(f'Missing climate input: {p}')
        d=norm(pd.read_csv(p)); d['date']=pd.to_datetime(d.date,errors='coerce')
        d=d.dropna(subset=['date'])
        if 'scenario' not in d: raise ValueError(f'{p} lacks a Scenario column')
        d=d[d.scenario.astype(str).str.lower().eq(scenario)].copy()
        if 'period' in d:
            d['period']=d.period.astype(str).str.lower().str.replace('term','',regex=False)
        return d
    outputs=[]
    for scenario in ['ssp245','ssp585']:
        precip=[]
        for model in models:
            d=climate(model,scenario)
            if 'avg_pcp' not in d: raise ValueError(f'{model}/{scenario} needs precipitation column avg_pcp or p_mm')
            # NASA GDDP-CMIP6 precipitation is kg m-2 s-1; convert to mm/day.
            pcp=d.avg_pcp.astype(float)
            if pcp.median() < 0.1: pcp=pcp*86400.0
            precip.append(pd.DataFrame({'date':d.date,'pcp_'+model:pcp}))
        p=precip[0]
        for q in precip[1:]: p=p.merge(q,on='date',how='inner')
        p['avg_pcp']=p[['pcp_'+m for m in models]].mean(axis=1)
        t=climate('CanESM5',scenario)
        for c in ['avg_tmax','avg_tmin']:
            if c not in t: raise ValueError(f'CanESM5/{scenario} requires tasmax/tasmin')
            if t[c].median()>100: t[c]=t[c]-273.15
        tcols=['date','avg_tmax','avg_tmin']+(['period'] if 'period' in t else [])
        d=p[['date','avg_pcp']].merge(t[tcols],on='date').sort_values('date').reset_index(drop=True)
        if d.empty: raise ValueError(f'No date overlap for {scenario} climate/QSWAT+ inputs.')
        d['scenario']=scenario
        d.to_csv(OUT_DIR/f'cmip6_six_model_climate_ensemble_{scenario}.csv',index=False)
        qfile=OUT_DIR/f'qswat_future_{scenario}.csv'
        if not qfile.exists():
            qswat_future_discharge(scenario,qfile)
        if not qfile.exists():
            print(f'Climate ensemble saved for {scenario}, but no scenario-specific QSWAT+ discharge was found at {qfile}; skipping hybrid flow projection for this scenario.')
            continue
        q=norm(pd.read_csv(qfile)); q['date']=pd.to_datetime(q.date,errors='coerce')
        d=d.merge(q[['date','Q_Simulated']],on='date').sort_values('date').reset_index(drop=True)
        if d.empty: raise ValueError(f'No date overlap for {scenario} climate and future QSWAT+ data.')
        flow=list(h.observed_flow.tail(7).astype(float)); pcp=list(h.avg_pcp.tail(7).astype(float)); simlag=float(h.Q_Simulated.iloc[-1]); forecasts=[]
        for step,(_,r) in enumerate(d.iterrows(),start=1):
            z={'Q_Simulated':float(r.Q_Simulated),'avg_pcp':float(r.avg_pcp),'avg_tmax':float(r.avg_tmax),'avg_tmin':float(r.avg_tmin),
              'avg_slr':float(h.avg_slr.mean()),'avg_wnd':float(h.avg_wnd.mean()),'avg_hmd':float(h.avg_hmd.mean()),
              'observed_flow_lag1':flow[-1],'simulated_flow_lag1':simlag,'avg_pcp_lag1':pcp[-1],'observed_flow_lag7':flow[0],
              'avg_pcp_lag7':pcp[0],'year':r.date.year,'month':r.date.month,'day_of_year':r.date.dayofyear,
              'day_of_week':r.date.dayofweek,'week_of_year':int(r.date.isocalendar().week),
              'rolling_avg_pcp_7d':np.mean(pcp),'rolling_avg_observed_flow_7d':np.mean(flow)}
            xx=pd.DataFrame([z])[FEATURES]; xs=scaler.transform(xx)
            b=pd.DataFrame({'QSWAT_Pred':[z['Q_Simulated']],'RF_Pred':rf.predict(xx),'XGB_Pred':xgb.predict(xx),
              'LightGBM_Pred':lgbm.predict(xx),'LSTM_Pred':[predict_lstm_single_step(lstm,xs)]})
            yhat=max(0.,float(meta.predict(b)[0])); forecasts.append(yhat)
            flow=(flow+[yhat])[-7:]; pcp=(pcp+[float(r.avg_pcp)])[-7:]; simlag=float(r.Q_Simulated)
            if step%3650==0: print(f'{scenario}: forecasted {step:,}/{len(d):,} days',flush=True)
        d['Hybrid_Predicted_Flow']=forecasts; d['scenario']=scenario; outputs.append(d)
        d.to_csv(OUT_DIR/f'future_cmip6_hybrid_streamflow_{scenario}.csv',index=False)
        print(f'Saved {scenario} daily hybrid flow after {len(d):,} forecast steps.',flush=True)
    if not outputs:
        print('CMIP6 climate ensembles are ready. Add real future QSWAT+ discharge series for both SSP scenarios to generate the QSWAT+-inclusive hybrid streamflow projection.')
        return
    future=pd.concat(outputs,ignore_index=True); future.to_csv(OUT_DIR/'future_cmip6_hybrid_streamflow.csv',index=False)
    for scen,g in future.groupby('scenario'): plt.plot(g.date,g.Hybrid_Predicted_Flow,label=scen)
    plt.legend(); plt.ylabel('Hybrid streamflow (m³/s)'); plt.xlabel('Date'); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(OUT_DIR/'future_cmip6_streamflow.png',dpi=180); plt.close()
    print('Saved future projections to',OUT_DIR.resolve())

def qswat_future_discharge(scenario,output_path):
    """Extract daily outlet flow from the user's near/mid SWAT+ scenario runs.

    SSPN folders cover 2015-2044. SSPM folders overlap from 2039, so only
    2045-2074 is used. cha001/gis_id 1 has the largest mean flo_out in the
    supplied channel monthly outputs and is used as the basin outlet series.
    SWAT+ flo_out is in m3/s.
    """
    folder_root=QSWAT_FUTURE_DIR
    suffix='245' if scenario=='ssp245' else '585'
    sources=[(folder_root/f'SSPN{suffix}',2015,2044),
             (folder_root/f'SSPM{suffix}',2045,2074)]
    parts=[]
    for folder,year_start,year_end in sources:
        db=folder/'Results'/'swatplus_output.sqlite'
        if not db.exists():
            raise FileNotFoundError(f'Missing QSWAT+ SQLite output: {db}')
        with sqlite3.connect(db) as con:
            q=pd.read_sql_query(
                "SELECT jday, mon, day, yr, flo_out FROM channel_sd_day "
                "WHERE gis_id=1 AND lower(trim(name))='cha001' AND yr BETWEEN ? AND ?",
                con,params=(year_start,year_end))
        if q.empty:
            raise ValueError(f'No daily cha001 flow rows in {db} for {year_start}-{year_end}.')
        q['date']=pd.to_datetime(dict(year=q.yr,month=q.mon,day=q.day),errors='coerce')
        q=q.dropna(subset=['date']).rename(columns={'flo_out':'Q_Simulated'})
        parts.append(q[['date','Q_Simulated']])
    out=pd.concat(parts,ignore_index=True).sort_values('date').drop_duplicates('date')
    if out.Q_Simulated.isna().any():
        raise ValueError(f'Invalid daily QSWAT+ values for {scenario}.')
    out.to_csv(output_path,index=False)
    print(f'Extracted {len(out):,} daily {scenario} QSWAT+ outlet flows '
          f'({out.date.min().date()} to {out.date.max().date()}; cha001).')

if __name__ == '__main__':
    fitted=run()
    run_cmip6(*fitted)
